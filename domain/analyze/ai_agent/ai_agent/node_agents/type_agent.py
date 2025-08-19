# type_agent.py  (Gemini 2.0 Flash 버전; 영어 카테고리 반환)
import os, io, json, base64
from dotenv import load_dotenv
from PIL import Image
import google.generativeai as genai

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

def _gemini_img_part(img_bytes: bytes) -> dict:
    """바이트 이미지를 Gemini용 포맷으로 변환"""
    return {"mime_type": "image/jpeg", "data": img_bytes}

class TypeAgent:
    def __init__(self, model: str = "gemini-2.5-flash-lite"):
        self.model = model
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("환경변수 GOOGLE_API_KEY가 필요합니다.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    def analyze(self, front_image: bytes, left_image: bytes, rear_image: bytes, right_image: bytes):
        # 4개 이미지를 분석하여 종류/건전지/크기 판별
        prompt = """
You are a toy category expert. Look at four views (front/left/rear/right) and output STRICT JSON only.
Category must be one of:
{"type":"robot|building_blocks|dolls|vehicles|educational|action_figures|board_games|musical|sports|others","battery":"건전지|비건전지|불명","size":"작음|중간|큄|불명"}

Rules:
- If Korean words like '모형' appear, map it to 'action_figures' (category list above).
- '건전지 장난감/비건전지 장난감' is not a type; put in 'battery'.
- Size is relative: 작음/중간/큄 (uncertain → '불명').
- STRICT JSON only. No extra text.
""".strip()

        try:
            # 이미지 준비
            images = []
            for img_bytes in [front_image, left_image, rear_image, right_image]:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                images.append(img)

            # Gemini 호출
            response = self.model.generate_content([prompt] + images)
            
            # 토큰 정보 추출
            token_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            try:
                if hasattr(response, 'usage_metadata'):
                    token_info.update({
                        "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                        "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0),
                        "total_tokens": getattr(response.usage_metadata, 'total_token_count', 0)
                    })
            except Exception:
                pass

            raw = response.text.strip()
            if raw.startswith("```json"):
                raw = raw[7:]
            if raw.startswith("```"):
                raw = raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            result = raw.strip()

            if not result:
                result = '{"type":"others","battery":"불명","size":"불명"}'

            print(f"TypeAgent 응답: {result} (토큰: {token_info['total_tokens']})")
            return result, token_info

        except Exception as e:
            print(f"❌ TypeAgent 에러: {e}")
            import traceback
            traceback.print_exc()
            return '{"type":"others","battery":"불명","size":"불명"}', {"total_tokens": 0}
