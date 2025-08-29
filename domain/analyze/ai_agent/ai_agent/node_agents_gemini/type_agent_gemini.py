# type_agent_gemini.py  (Google Gemini, single-image; returns "robot" 등 문자열만)
import os, io, base64
from dotenv import load_dotenv
from PIL import Image
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, TimeoutError

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

class TypeAgent:
    def __init__(self):
        self.model = "gemini-2.5-flash-lite"
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("환경변수 GEMINI_API_KEY 필요")
        genai.configure(api_key=api_key)
        self.timeout = 30  # 30초 타임아웃

        self.prompt = """
You are a toy category expert. One toy image will be provided.

Answer ONLY with the category in English:
robot | building_blocks | dolls | vehicles | educational | action_figures | board_games | musical | sports | others

Do not include JSON, notes, or extra text. Just output one word.
""".strip()

        self._model = genai.GenerativeModel(self.model)

    def analyze(self, image_bytes: bytes) -> str:
        try:
            if image_bytes is None:
                return "others"

            b64 = _img_bytes_to_b64jpeg(image_bytes)
            img_part = {"mime_type": "image/jpeg", "data": base64.b64decode(b64)}

            # ThreadPoolExecutor를 사용하여 타임아웃 처리
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._analyze_with_gemini, img_part)
                resp = future.result(timeout=self.timeout)

            text = (getattr(resp, "text", None) or "").strip().lower()

            valid = [
                "robot","building_blocks","dolls","vehicles",
                "educational","action_figures","board_games",
                "musical","sports","others"
            ]

            for v in valid:
                if v in text:
                    return v

            return "others"

        except TimeoutError:
            print("TypeAgent: API 호출 타임아웃, 기본값 others 반환")
            return "others"
        except Exception as e:
            print(f"TypeAgent error: {e}")
            return "others"

    def _analyze_with_gemini(self, img_part):
        """Gemini API 호출을 별도 메서드로 분리"""
        return self._model.generate_content(
            [self.prompt, img_part],
            generation_config={"temperature": 0.0},
            safety_settings=None,
        )
