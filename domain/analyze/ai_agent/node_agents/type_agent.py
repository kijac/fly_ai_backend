import os
import base64
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class TypeAgent:
    def __init__(self, model="gpt-4o"):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def analyze(self, image_bytes):
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "당신은 장난감 종류 구별 전문가 입니다. 이 이미지의 장난감을 분석해주세요:\n1. 종류: 인형, 피규어, 도서, 모형, 블록, 자동차, 변신로봇, 퍼즐, 보드게임 중 하나\n2. 건전지 사용 여부: 건전지 또는 비건전지\n\n반드시 다음 JSON 형식으로만 답변하세요:\n{\"type\": \"종류\", \"battery\": \"건전지여부\"}"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}",
                                    "detail": "low"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=150,
                temperature=0.1
            )
            result = response.choices[0].message.content.strip()
            total_tokens = getattr(response, 'usage', None)
            total_tokens = total_tokens.total_tokens if total_tokens and hasattr(total_tokens, 'total_tokens') else None
            print(f"TypeAgent raw response: {result}")
            return result, total_tokens
        except Exception as e:
            print(f"TypeAgent 에러: {e}")
            return '{"type": "피규어", "battery": "비건전지"}', None
