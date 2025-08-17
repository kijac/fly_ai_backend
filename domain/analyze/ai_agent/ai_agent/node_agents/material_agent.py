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

class MaterialAgent:
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("환경변수 ANTHROPIC_API_KEY가 필요합니다.")
        self.client = Anthropic(api_key=api_key)

    def analyze(self, front_image: bytes, left_image: bytes, rear_image: bytes, right_image: bytes):
        # 4개 이미지를 분석하여 재료 판별 (출력 스키마/반환값은 기존과 동일)
        system_prompt = """
당신은 장난감 재료 분석 전문가입니다.
앞면, 왼쪽, 뒷면, 오른쪽 4개 각도 이미지를 종합해 재료를 판별하세요.

오직 아래 JSON만 반환하세요(마크다운 금지):
{"material":"재료","material_detail":"단일 소재|혼합 소재","confidence":"높음|보통|낮음","notes":"상세설명"}

분류 가이드:
- 단일 소재(확신할 때만): 플라스틱/금속/나무/섬유(천)/실리콘/유리/고무
- 혼합 소재(의심되면 즉시): 예) 플라스틱,금속 / 플라스틱,섬유 / 플라스틱,고무 ...
주의: 금속(나사/축), 섬유(옷/털), 실리콘/고무 파트가 보이면 '혼합 소재'로.
불확실하면 '혼합 소재' 또는 confidence='낮음'.
""".strip()

        try:
            # 이미지 4장 base64 준비
            b64s = []
            for img_bytes in [front_image, left_image, rear_image, right_image]:
                b64s.append(_img_bytes_to_b64jpeg(img_bytes))

            contents = [{"type": "text", "text": system_prompt},
                        {"type": "text", "text": "아래 4개 각도를 참고해 재료를 판별하세요."}]
            for b in b64s:
                contents.append(_anthropic_img_part(b))

            # Claude 호출
            msg = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                temperature=0.0,
                system="Return STRICT JSON only. No extra text.",
                messages=[{"role": "user", "content": contents}]
            )

            # usage → token_info (기존 키와 호환되게 매핑)
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
                result = '{"material":"플라스틱","material_detail":"단일 소재","confidence":"낮음","notes":"빈 응답으로 기본값"}'

            print(f"MaterialAgent 응답: {result} (토큰: {token_info['total_tokens']})")
            return result, token_info

        except Exception as e:
            print(f"MaterialAgent 에러: {e}")
            return '{"material":"플라스틱","material_detail":"단일 소재","confidence":"낮음","notes":"분석 실패 기본값"}', {"total_tokens": 0}
