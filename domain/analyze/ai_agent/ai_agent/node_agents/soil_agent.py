# soil_agent.py  (Claude 버전, 기준 1장, 절대점수/상대등급 A~E 반환)
import os, io, json, base64
from typing import Tuple, Dict

from PIL import Image
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

def _img_bytes_to_b64jpeg(img_bytes: bytes, max_side: int = 1024) -> str:
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

# 깨끗함 절대점수: 4(새것 수준) → 0(매우 더러움)
_SOIL_LABEL_TO_ABS = {
    "깨끗": 4, "보통": 3, "약간 더러움": 2, "더러움": 1, "판단 불가": 2
}

def _label_to_abs(s: str) -> int:
    s = (s or "").strip()
    for k, v in _SOIL_LABEL_TO_ABS.items():
        if k in s:
            return v
    return 2

def _steps_to_grade(steps: int) -> str:
    # steps = reference_abs - query_abs (신품대비 얼마나 덜 깨끗한지)
    if steps <= 0: return "A"
    if steps == 1: return "B"
    if steps == 2: return "C"
    if steps == 3: return "D"
    return "E"

class SoilAgent:
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

        # 5) 절대점수/상대등급 산출
        q_abs = _label_to_abs(base.get("query", {}).get("soil"))
        r_abs = _label_to_abs(base.get("reference", {}).get("soil"))
        steps = r_abs - q_abs          # 신품대비 몇 단계 낮은지(>0이면 사용자쪽이 더 오염)
        grade = _steps_to_grade(steps) # A~E

        result = {
            "query": base.get("query", {}),
            "reference": base.get("reference", {}),
            "query_abs": int(q_abs),
            "reference_abs": int(r_abs),
            "grade": grade,  # 상대등급
            "relative_vs_ref": {"delta": int(q_abs - r_abs), "grade": grade}  # delta는 로그 호환(-1 등)
        }

        # (디버그) 콘솔 로그
        try:
            print(f"SoilAgent result: {json.dumps(result, ensure_ascii=False)} | tokens={usage['total_tokens']}")
        except Exception:
            pass

        return result, usage
