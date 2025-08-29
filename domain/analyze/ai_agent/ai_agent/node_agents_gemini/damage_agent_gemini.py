# ai_agent/node_agents_gemini/damage_agent_gemini.py
# - Compare REF(new) vs USED in one call (Gemini 2.5 Flash/Lite)
# - Output ONLY grade (A~E)

import os, io, json, base64, time
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, TimeoutError

load_dotenv()

# ----------------- helpers -----------------
def _img_to_b64jpeg(path_or_file, max_side: int = 1024) -> str:
    try:
        img = Image.open(path_or_file).convert("RGB")
    except Exception:
        img = Image.new("RGB", (256, 256), "white")
    w, h = img.size
    s = min(1.0, max_side / max(w, h))
    if s < 1.0:
        img = img.resize((int(w*s), int(h*s)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")



_PROMPT = """
You are a precise toy DAMAGE judge.
Compare two images: REF (new toy) vs USED (secondhand toy).
Decide ONLY the USED toy's damage grade (A-E):
- A = 없음 (looks like new; very clean, no noticeable wear; all major parts present)
- B = 미세한 파손 (tiny scratch, slight dust, very small wear; all parts intact)
- C = 경미한 파손 (visible scratches, paint wear, small cracks, or minor deformation; all parts intact)
- D = 부품 누락 (ANY major part visible in REF is missing in USED: especially arms, legs, head, wheel, rotor, weapon, shield, panel)
- E = 심각한 파손 (major break, arm/leg/head broken off, torso cracked, multiple major parts missing, severe deformation)
STRICT POLICY:
- STEP 1: Always check for arms and legs FIRST.
  • If REF shows a left/right arm or leg but USED is missing it → grade = "D".
  • If an arm/leg is clearly broken/detached → grade = "E".
- STEP 2: Then check other major parts (head, torso, wheels, rotors, wings, weapons, shields, panels).
  • Missing part → "D".
  • Broken/detached → "E".
- Only if ALL major parts are intact, then evaluate surface condition:
  • Looks like new → "A"
  • Tiny wear/dust → "B"
  • Noticeable wear/minor cracks → "C"
- "A", "B", "C" may be given more leniently if toy is complete.
- Minor surface differences, tiny scratches, or very small color/dust variations → still allow "A" or "B".
- When uncertain:
  • Between A/B/C → choose the milder (better) grade.
  • Between C/D/E → choose the stricter (worse) grade.
DETERMINISM REQUIREMENT:
- Your judgment MUST be deterministic.
- The SAME REF and USED images MUST ALWAYS produce the SAME grade, every time.
- Do not use randomness, variation, or different interpretations across calls.
Return STRICT JSON ONLY:
{"grade":"A" | "B" | "C" | "D" | "E"}
No explanations. JSON only.
""".strip()


# ----------------- DamageAgent class -----------------
class DamageAgent:
    # def __init__(self, model: str = "gemini-2.5-flash-lite"):
    def __init__(self, model: str = "gemini-2.5-flash"):
    
        
        api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.GenerativeModel(model)
        self.model = model
        self.timeout = 30  # 30초 타임아웃

    def analyze(self, ref_image_bytes: bytes, used_image_bytes: bytes) -> str:
        """Compare REF vs USED images -> return grade (A~E, string only)"""
        try:
            ref_b64 = base64.b64encode(ref_image_bytes).decode("utf-8")
            used_b64 = base64.b64encode(used_image_bytes).decode("utf-8")

            # ThreadPoolExecutor를 사용하여 타임아웃 처리
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._analyze_with_gemini, ref_b64, used_b64)
                resp = future.result(timeout=self.timeout)

            try:
                out = json.loads(resp.text.strip())
                grade = out.get("grade", "C")
            except Exception:
                grade = "C"  # fallback

            return grade
            
        except TimeoutError:
            print("DamageAgent: API 호출 타임아웃, 기본값 C 반환")
            return "C"
        except Exception as e:
            print(f"DamageAgent 에러: {e}, 기본값 C 반환")
            return "C"

    def _analyze_with_gemini(self, ref_b64: str, used_b64: str):
        """Gemini API 호출을 별도 메서드로 분리"""
        return self.client.generate_content(
            contents=[{
                "role": "user",
                "parts": [
                    {"text": _PROMPT},
                    {"inline_data": {"mime_type": "image/jpeg", "data": ref_b64}},
                    {"inline_data": {"mime_type": "image/jpeg", "data": used_b64}},
                ]
            }],
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.0,   # ★ 항상 같은 결과를 강제
                "top_p": 1.0,         # (선택) 확률 샘플링 억제
                "top_k": 1            # (선택) 후보 수 제한
            }
        )
