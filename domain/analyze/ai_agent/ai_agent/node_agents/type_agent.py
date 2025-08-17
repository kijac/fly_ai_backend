# type_agent.py  (Claude 버전)
import os, io, json, base64
from dotenv import load_dotenv
from PIL import Image
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

def _anthropic_img_part(b64: str) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}

class TypeAgent:
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("환경변수 ANTHROPIC_API_KEY가 필요합니다.")
        self.client = Anthropic(api_key=api_key)

    def analyze(self, front_image: bytes, left_image: bytes, rear_image: bytes, right_image: bytes):
        # 4개 이미지를 분석하여 종류/건전지/크기 판별
        system_prompt = """
당신은 장난감 종류 및 특성 분석 전문가입니다.
앞/좌/후/우 4개 각도 이미지를 종합해 아래 JSON만 반환하세요. (마크다운 금지)

{"type":"종류","battery":"건전지|비건전지|불명","size":"작음|중간|큄|불명"}

규칙:
- type 후보: 피규어, 자동차 장난감, 변신 로봇, 인형, 블록, 공, 아동 도서, 플라스틱 부품, 나무 장난감, 보행기, 탈것, 기타
- '모형'이라는 표현은 최종 출력에서 반드시 '피규어'로 통일.
- '건전지 장난감/비건전지 장난감'은 type이 아니라 battery 필드로 구분.
- 크기는 상대적 시각 단서(손/바닥 타일/문 손잡이 등)를 이용: 작음/중간/큄 중 고르되 애매하면 '불명'.
- 확신이 낮으면 battery='불명', size='불명'으로.
- STRICT JSON only. No extra text.
""".strip()

        try:
            # 이미지 4장 base64 준비
            b64s = []
            for img_bytes in [front_image, left_image, rear_image, right_image]:
                b64s.append(_img_bytes_to_b64jpeg(img_bytes))

            contents = [{"type": "text", "text": system_prompt},
                        {"type": "text", "text": "분석 대상 4개 각도 이미지:"}]
            for b in b64s:
                contents.append(_anthropic_img_part(b))

            # Claude 호출
            msg = self.client.messages.create(
                model=self.model,
                max_tokens=200,
                temperature=0.0,
                system="Return STRICT JSON only. No extra text.",
                messages=[{"role": "user", "content": contents}]
            )

            # usage → token_info 매핑
            token_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            try:
                u = getattr(msg, "usage", None)
                itok = int(getattr(u, "input_tokens", 0)) if u else 0
                otok = int(getattr(u, "output_tokens", 0)) if u else 0
                token_info.update({
                    "prompt_tokens": itok,
                    "completion_tokens": otok,
                    "total_tokens": itok + otok
                })
            except Exception:
                pass

            # 텍스트 추출 & 코드펜스 제거
            raw = "".join(b.text for b in msg.content if hasattr(b, "text")).strip()
            if raw.startswith("```json"):
                raw = raw[7:]
            if raw.startswith("```"):
                raw = raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            result = raw.strip()

            if not result:
                result = '{"type":"기타","battery":"불명","size":"불명"}'

            print(f"TypeAgent 응답: {result} (토큰: {token_info['total_tokens']})")
            return result, token_info

        except Exception as e:
            print(f"TypeAgent 에러: {e}")
            return '{"type":"기타","battery":"불명","size":"불명"}', {"total_tokens": 0}
