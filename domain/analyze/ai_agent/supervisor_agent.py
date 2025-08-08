from .node_agents.type_agent import TypeAgent
from .node_agents.material_agent import MaterialAgent
from .node_agents.damage_agent import DamageAgent
import json

class SupervisorAgent:
    def __init__(self):
        self.type_agent = TypeAgent()
        self.material_agent = MaterialAgent()
        self.damage_agent = DamageAgent()

    def process(self, image_bytes):
        # 1. 각 노드 에이전트로부터 정보 수집 및 토큰 카운트
        print("TypeAgent 분석 중...")
        type_response, type_tokens = self.type_agent.analyze(image_bytes)
        print(f"TypeAgent 응답: {type_response}")

        print("MaterialAgent 분석 중...")
        material_response, material_tokens = self.material_agent.analyze(image_bytes)
        print(f"MaterialAgent 응답: {material_response}")

        print("DamageAgent 분석 중...")
        damage_response, damage_tokens = self.damage_agent.analyze(image_bytes)
        print(f"DamageAgent 응답: {damage_response}")

        # JSON 파싱을 안전하게 처리
        try:
            type_result = json.loads(type_response)
        except json.JSONDecodeError:
            print(f"TypeAgent JSON 파싱 실패: {type_response}")
            type_result = {"type": "알 수 없음", "battery": "알 수 없음"}

        try:
            material_result = json.loads(material_response)
        except json.JSONDecodeError:
            print(f"MaterialAgent JSON 파싱 실패: {material_response}")
            material_result = {"material": "알 수 없음"}

        try:
            damage_result = json.loads(damage_response)
        except json.JSONDecodeError:
            print(f"DamageAgent JSON 파싱 실패: {damage_response}")
            damage_result = {"damage": "알 수 없음"}

        toy_type = type_result.get("type", "")
        battery = type_result.get("battery", "")
        material = material_result.get("material", "")
        damage = damage_result.get("damage", "")

        # 2. 기부 가능 여부 판별
        donate, reason = self.judge_donation(toy_type, battery, material)
        # 3. 수리/분해 여부 판별
        repair_or_disassemble = self.judge_repair(damage)

        # 토큰 정보 dict로 반환
        token_usage = {
            "type_agent": type_tokens,
            "material_agent": material_tokens,
            "damage_agent": damage_tokens,
            "total": sum([t for t in [type_tokens, material_tokens, damage_tokens] if t is not None])
        }

        return {
            "장난감 종류": toy_type,
            "건전지 여부": battery,
            "재료": material,
            "파손": damage,
            "기부 가능 여부": "가능" if donate else "불가능",
            "기부 불가 사유": reason if not donate else None,
            "수리/분해": repair_or_disassemble,
            "토큰_사용량": token_usage
        }

    def judge_donation(self, toy_type, battery, material):
        # 기부 가능/불가 로직 (요구사항 반영)
        donate_types = ["인형", "피규어", "블록", "자동차", "변신로봇", "퍼즐", "보드게임", "모형"]
        prefer_material = "플라스틱"
        not_donate_types = ["도서", "아동도서"]
        
        # 기부 불가 종류 체크
        if toy_type in not_donate_types:
            return False, f"기부 불가 종류({toy_type})"
        
        # 나무 소재 체크
        if material == "나무":
            return False, "나무 소재는 기부 불가"
            
        # 섬유 재질 체크 (새로 추가)
        if material == "섬유":
            return False, "섬유 재질은 기부 불가"
            
        # 기부 가능 종류이거나 플라스틱 소재면 기부 가능
        if toy_type in donate_types or material == prefer_material:
            return True, None
            
        return False, "기준에 부합하지 않음"

    def judge_repair(self, damage):
        # 파손 정보 기반 수리/분해 판별
        if damage == "없음":
            return "수리 불필요(완제품)"
        elif "심각" in damage or "크게" in damage:
            return "분해 필요"
        else:
            return "수리 가능"
