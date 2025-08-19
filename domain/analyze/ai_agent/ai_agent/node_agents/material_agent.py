# material_agent.py  (Claude 버전, 영어 카테고리 출력, mixed는 확실할 때만)
import os, io, json, base64
from typing import Tuple, Dict, Optional, List

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

class MaterialAgent:
    """
    결과는 영어로:
      material: "plastic|metal|wood|fabric|silicone|rubber|paper_cardboard|electronic|mixed|unknown"
      components: ["plastic","metal"]  # material == "mixed"일 때만
      confidence: 0.0~1.0
      material_detail: 짧은 이유
      notes: 선택
      secondary_hint: 혼합으로 확정하기 애매할 때 약한 2순위 후보(없으면 null)
    """
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("환경변수 ANTHROPIC_API_KEY가 필요합니다.")
        self.client = Anthropic(api_key=api_key)

        self.prompt = """
You are a MATERIAL classifier for toys.

Return STRICT JSON ONLY (no markdown, no prose):
{
  "material": "plastic|metal|wood|fabric|silicone|rubber|paper_cardboard|electronic|mixed|unknown",
  "components": ["plastic","metal"],           // only when material=="mixed", 2~3 items max
  "secondary_hint": "plastic|metal|wood|fabric|silicone|rubber|paper_cardboard|electronic|null",
  "confidence": 0.0,                           // 0.0~1.0 overall confidence
  "material_detail": "...",                    // short reason (<= 120 chars)
  "notes": "..."                               // optional short notes
}

Guidelines:
- OUTPUT IN ENGLISH.
- Be CONSERVATIVE with "mixed": choose "mixed" ONLY when you clearly see at least two distinct materials across different parts/surfaces
  (not just reflections, paint, decals, or lighting artifacts). Require solid visual cues (texture, seams, shine, deformation).
- If you suspect a second material but evidence is weak, DO NOT output "mixed". Instead:
  - choose the single DOMINANT material for "material"
  - and set "secondary_hint" to the weak candidate (otherwise null).
- Prefer "plastic" for typical injection-molded toy bodies unless strong evidence contradicts.
- "fabric" includes cloth/textiles; "paper_cardboard" includes paper, carton; "electronic" is for visible PCBs/solder/ICs dominating the object.
- Keep responses short. STRICT JSON only.
""".strip()

    def analyze(self,
                front_image: bytes, left_image: bytes, rear_image: bytes, right_image: bytes):
        # 이미지 4장 준비
        b64s = []
        for img_bytes in [front_image, left_image, rear_image, right_image]:
            b64s.append(_img_bytes_to_b64jpeg(img_bytes))

        contents = [{"type": "text", "text": self.prompt},
                    {"type": "text", "text": "4-view images (front/left/rear/right):"}]
        for b in b64s:
            contents.append(_anthropic_img_part(b))

        try:
            msg = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                temperature=0.0,
                system="Return STRICT JSON only.",
                messages=[{"role": "user", "content": contents}]
            )

            usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            try:
                u = getattr(msg, "usage", None)
                itok = int(getattr(u, "input_tokens", 0)) if u else 0
                otok = int(getattr(u, "output_tokens", 0)) if u else 0
                usage.update({"input_tokens": itok, "output_tokens": otok, "total_tokens": itok + otok})
            except Exception:
                pass

            raw = "".join(b.text for b in msg.content if hasattr(b, "text")).strip()
            if raw.startswith("```json"):
                raw = raw[7:]
            if raw.startswith("```"):
                raw = raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            result = raw.strip()

            # 비정형/빈 응답 방어
            try:
                parsed = json.loads(result) if result else {}
            except Exception:
                parsed = {}

            # 필드 보정
            material = parsed.get("material") or "unknown"
            if material != "mixed":
                parsed["components"] = []
            if "secondary_hint" not in parsed:
                parsed["secondary_hint"] = None
            if "confidence" not in parsed:
                parsed["confidence"] = 0.0
            if "material_detail" not in parsed:
                parsed["material_detail"] = ""
            if "notes" not in parsed:
                parsed["notes"] = ""

            out_json = json.dumps(parsed, ensure_ascii=False)
            print(f"MaterialAgent 응답: {out_json} | tokens={usage['total_tokens']}")
            return out_json, usage

        except Exception as e:
            print(f"MaterialAgent 에러: {e}")
            fallback = {
                "material": "unknown",
                "components": [],
                "secondary_hint": None,
                "confidence": 0.0,
                "material_detail": "fallback",
                "notes": ""
            }
            return json.dumps(fallback, ensure_ascii=False), {"total_tokens": 0}
