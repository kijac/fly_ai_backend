import os
import base64
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class DamageAgent:
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
                                "text": "당신은 장난감 파손여부판별 전문가 입니다. 이 장난감의 파손 여부를 분석해주세요:\n상태: 없음, 미세한 파손, 심각한 파손 중 하나\n\n반드시 다음 JSON 형식으로만 답변하세요:\n{\"damage\": \"상태\"}"
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
                max_tokens=100,
                temperature=0.1
            )
            result = response.choices[0].message.content.strip()
            total_tokens = getattr(response, 'usage', None)
            total_tokens = total_tokens.total_tokens if total_tokens and hasattr(total_tokens, 'total_tokens') else None
            print(f"DamageAgent raw response: {result}")
            return result, total_tokens
        except Exception as e:
            print(f"DamageAgent 에러: {e}")
            return '{"damage": "없음"}', None
