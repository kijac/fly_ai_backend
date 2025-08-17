# soil_agent.py  (Claude 버전, 기준 이미지 1장, 5단계 상대판정)
import os, io, json, base64
from typing import Tuple, Dict

from PIL import Image
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

def _img_bytes_to_b64jpeg(img_bytes: bytes, max_side: int = 1024) -> str:
    """바이트 → RGB → 리사이즈 → JPEG → base64"""
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        img = Image.new("RGB", (256, 256), color="white")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _anthropic_img_part(b64: str) -> Dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}

def _soil_to_int(s: str) -> int:
    """
    오염 라벨 → 정수(클수록 '더 더러움').
    """
    t = (s or "").strip()
    if "깨끗" in t: return 0
    if "보통" in t: return 1
    if "약간" in t and "더러움" in t: return 2   # '약간 더러움'
    if "더러움" in t: return 3                   # 심한 오염
    if "판단" in t: return 2                     # 판단 불가 → 중간값으로
    return 2

class SoilAgent:
    """
    Claude 멀티모달.
    입력: 사용자 4뷰(앞/좌/후/우) + 기준 1장(ref_image)
    출력: query/ref 오염 라벨 + 5단계 상대 레벨(soil_delta_level: -2..+2)
    """
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("환경변수 ANTHROPIC_API_KEY가 필요합니다.")
        self.client = Anthropic(api_key=api_key)

        self.prompt = """
You are a toy CLEANLINESS (soil/dirt) judge.
Analyze QUERY (front/left/rear/right) vs a single REFERENCE image.

Return STRICT JSON ONLY:
{
  "query":    {"soil":"깨끗|보통|약간 더러움|더러움|판단 불가","soil_detail":"..."},
  "reference":{"soil":"깨끗|보통|약간 더러움|더러움|판단 불가","soil_detail":"..."}
}
Guidelines:
- '깨끗' = 거의 오염 없음. '보통' = 약간의 사용흔적. '약간 더러움' = 눈에 띄는 얼룩/먼지. '더러움' = 심한 오염/위생 문제.
- If unsure, prefer '판단 불가' over optimistic '깨끗'.
- JSON only. No extra text.
""".strip()

    def analyze(self,
                query_front: bytes, query_left: bytes, query_rear: bytes, query_right: bytes,
                ref_image: bytes) -> Tuple[dict, dict]:
        """
        ref_image: 기준 이미지 1장 (홀수=중앙값 대응, 짝수=유사도 1위는 바깥 로직에서 선택)
        return: (result_json(dict), usage(dict))
        """
        # 1) 이미지 준비
        q_b64s = [
            _img_bytes_to_b64jpeg(query_front),
            _img_bytes_to_b64jpeg(query_left),
            _img_bytes_to_b64jpeg(query_rear),
            _img_bytes_to_b64jpeg(query_right),
        ]
        r_b64 = _img_bytes_to_b64jpeg(ref_image)

        contents = [{"type": "text", "text": self.prompt},
                    {"type": "text", "text": "QUERY IMAGES (front/left/rear/right):"}]
        for b in q_b64s:
            contents.append(_anthropic_img_part(b))
        contents.extend([
            {"type": "text", "text": "REFERENCE IMAGE:"},
            _anthropic_img_part(r_b64)
        ])

        # 2) Claude 호출
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            temperature=0.0,
            system="Return STRICT JSON only.",
            messages=[{"role": "user", "content": contents}]
        )

        # 3) usage
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        try:
            u = getattr(msg, "usage", None)
            itok = int(getattr(u, "input_tokens", 0)) if u else 0
            otok = int(getattr(u, "output_tokens", 0)) if u else 0
            usage.update({"input_tokens": itok, "output_tokens": otok, "total_tokens": itok + otok})
        except Exception:
            pass

        # 4) JSON 파싱
        raw = "".join(b.text for b in msg.content if hasattr(b, "text")).strip()
        if raw.startswith("```"):
            raw = raw.strip("`").replace("json", "", 1).strip()
        try:
            base = json.loads(raw)
        except Exception as e:
            base = {
                "query": {"soil": "판단 불가", "soil_detail": f"parse_error: {e}"},
                "reference": {"soil": "판단 불가", "soil_detail": ""}
            }

        # 5) 5단계 상대 레벨 계산 (-2..+2)  ※ 값이 클수록 '더 깨끗'
        q = _soil_to_int(base.get("query", {}).get("soil"))
        r = _soil_to_int(base.get("reference", {}).get("soil"))
        d = r - q   # (양수) = 쿼리가 기준보다 '더 깨끗'

        if d >= 2: level = 2
        elif d == 1: level = 1
        elif d == 0: level = 0
        elif d == -1: level = -1
        else: level = -2  # d <= -2

        # ✅ 숫자만 반환하도록 변경 (라벨 제거)
        rel = {
            "soil_delta_level": int(level)   # -2..+2
        }

        result = {
            "soil_level": int(level),        # ✅ 최상위에도 숫자 레벨 제공
            "query": base.get("query", {}),
            "reference": base.get("reference", {}),
            "relative_vs_ref": rel
        }

        # (디버그) 콘솔 로그
        try:
            print(f"SoilAgent result: {json.dumps(result, ensure_ascii=False)} | tokens={usage['total_tokens']}")
        except Exception:
            pass

        return result, usage

