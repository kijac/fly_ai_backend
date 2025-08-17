import os, io, json, base64
from typing import List, Tuple, Optional, Dict

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

def _damage_to_int(s: str) -> int:
    """
    파손 라벨 → 정수(클수록 나쁨). 판단 불가는 중간값으로 취급.
    """
    t = str(s or "").strip()
    if "없음" in t: return 0
    if "미세" in t: return 1
    if "경미" in t: return 2
    if "부품 누락" in t: return 3
    if "심각" in t: return 4
    if "판단" in t: return 2
    return 2

def _mparts_to_bool(s: str) -> Optional[bool]:
    t = str(s or "").strip()
    if "있음" in t: return True
    if "없음" in t: return False
    return None  # 불명

class DamageAgent:
    """
    Claude 멀티모달. 입력: 사용자 4뷰(앞/좌/후/우) + 기준 1장(ref)
    결과: query/ref 라벨 + 5단계 상대 레벨(damage_delta_level: -2..+2)
    """
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("환경변수 ANTHROPIC_API_KEY가 없습니다.")
        self.client = Anthropic(api_key=api_key)

        # 비교용 프롬프트(간결/엄격 JSON)
        self.prompt = """
You are a toy DAMAGE and MISSING-PARTS judge.
Analyze QUERY (front/left/rear/right) vs a single REFERENCE image.

Return STRICT JSON ONLY:
{
  "query":    {"damage":"없음|미세한 파손|경미한 파손|부품 누락|심각한 파손|판단 불가","missing_parts":"없음|있음|불명","damage_detail":"..."},
  "reference":{"damage":"없음|미세한 파손|경미한 파손|부품 누락|심각한 파손|판단 불가","missing_parts":"없음|있음|불명","damage_detail":"..."}
}
Rules:
- Consider cracks, fractures, bent/warped parts, chips, base/joint detachment, severe scratches/paint loss.
- If unsure, be conservative: prefer '판단 불가' over optimistic '없음'.
- JSON only. No extra text.
""".strip()

    def analyze(self,
                query_front: bytes, query_left: bytes, query_rear: bytes, query_right: bytes,
                ref_image: bytes) -> Tuple[dict, dict]:
        """
        ref_image: 기준 이미지 1장(홀수=중앙값 대응, 짝수=유사도 1위 등은 바깥에서 결정)
        return: (result_json, usage)
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
        for b64 in q_b64s:
            contents.append(_anthropic_img_part(b64))
        contents.append({"type": "text", "text": "REFERENCE IMAGE:"})
        contents.append(_anthropic_img_part(r_b64))

        # 2) Claude 호출
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=400,
            temperature=0.0,
            system="Return STRICT JSON only.",
            messages=[{"role": "user", "content": contents}]
        )

        # 3) 토큰 usage
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
                "query":    {"damage":"판단 불가","missing_parts":"불명","damage_detail":f"parse_error: {e}"},
                "reference":{"damage":"판단 불가","missing_parts":"불명","damage_detail":""}
            }

        # 5) 5단계 상대 레벨 계산 (-2..+2)
        qd = _damage_to_int(base.get("query", {}).get("damage"))
        rd = _damage_to_int(base.get("reference", {}).get("damage"))
        d  = rd - qd  # 양수면 Query가 기준보다 '양호'

        # 기본 레벨
        if d >= 2: level = 2
        elif d == 1: level = 1
        elif d == 0: level = 0
        elif d == -1: level = -1
        else: level = -2  # d <= -2

        # 부품 누락 보정(강제 상/하향)
        qm = _mparts_to_bool(base.get("query", {}).get("missing_parts"))
        rm = _mparts_to_bool(base.get("reference", {}).get("missing_parts"))
        if qm is True and (rm is False or rm is None):
            level = min(level, -2)   # 사용자만 누락 → 최소 -2
        elif rm is True and (qm is False or qm is None):
            level = max(level,  2)   # 기준만 누락   → 최소 +2

        # ✅ 숫자만 반환하도록 변경
        rel = {
            "damage_delta_level": int(level)   # -2..+2 (라벨 제거)
        }

        result = {
            "damage_level": int(level),        # ✅ 최상위에도 숫자 레벨 제공
            "query": base.get("query", {}),
            "reference": base.get("reference", {}),
            "relative_vs_ref": rel
        }
        return result, usage