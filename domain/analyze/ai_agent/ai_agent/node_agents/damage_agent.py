# damage_agent.py  (Claude 버전, 기준 1장, 절대점수/상대등급 A~E 반환)
import os, io, json, base64
from typing import Tuple, Dict, Optional

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

# 손상 절대점수: 4(새것/손상없음) → 0(심각)
def _damage_to_abs(label: str) -> int:
    t = str(label or "").strip()
    if "없음" in t: return 4
    if "미세" in t: return 3
    if "경미" in t: return 2
    if "부품 누락" in t: return 1
    if "심각" in t: return 0
    if "판단" in t: return 2
    return 2

def _steps_to_grade(steps: int) -> str:
    # steps = reference_abs - query_abs (신품대비 얼마나 더 손상되었는지)
    if steps <= 0: return "A"
    if steps == 1: return "B"
    if steps == 2: return "C"
    if steps == 3: return "D"
    return "E"

class DamageAgent:
    """
    입력: 사용자 4뷰(앞/좌/후/우) + 기준 1장(ref)
    출력: query/ref 라벨, 절대점수(0..4), 상대등급(A~E)
    """
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("환경변수 ANTHROPIC_API_KEY가 없습니다.")
        self.client = Anthropic(api_key=api_key)

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
                "query":    {"damage":"판단 불가","missing_parts":"불명","damage_detail":f"parse_error: {e}"},
                "reference":{"damage":"판단 불가","missing_parts":"불명","damage_detail":""}
            }

        # 5) 절대점수/상대등급
        q_abs = _damage_to_abs(base.get("query", {}).get("damage"))
        r_abs = _damage_to_abs(base.get("reference", {}).get("damage"))
        steps = r_abs - q_abs
        grade = _steps_to_grade(steps)

        result = {
            "query": base.get("query", {}),
            "reference": base.get("reference", {}),
            "query_abs": int(q_abs),
            "reference_abs": int(r_abs),
            "grade": grade,
            "relative_vs_ref": {"steps": int(steps), "grade": grade},
            # 호환 키들
            "relative_damage_grade": grade,
            "query_damage_score": int(q_abs),
            "reference_damage_score": int(r_abs),
        }

        return result, usage
