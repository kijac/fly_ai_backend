# ai_agent/supervisor_agent.py

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from ai_agent.node_agents_gemini.damage_agent_gemini import DamageAgent
from ai_agent.node_agents_gemini.soil_agent_gemini import SoilAgent
from ai_agent.node_agents_gemini.type_agent_gemini import TypeAgent
from ai_agent.node_agents_gemini.material_agent_gemini import MaterialAgent

class SupervisorAgent:
    def __init__(self):
        self.damage = DamageAgent()
        self.soil = SoilAgent()
        self.type = TypeAgent()
        self.material = MaterialAgent()

    def process(self, ref_b: bytes, used_b: bytes) -> dict:
        """
        ref_b: 기준 이미지 (새 상품, bytes)
        used_b: 중고 이미지 (bytes)
        병렬로 4개 에이전트를 동시 실행하여 성능 향상
        """
        start_time = time.time()
        
        # 병렬 처리를 위한 ThreadPoolExecutor 사용
        with ThreadPoolExecutor(max_workers=4) as executor:
            # 각 에이전트를 동시에 실행
            futures = {
                executor.submit(self.damage.analyze, ref_b, used_b): "damage",
                executor.submit(self.soil.analyze, ref_b, used_b): "soil",
                executor.submit(self.type.analyze, used_b): "type",
                executor.submit(self.material.analyze, used_b): "material"
            }
            
            # 결과 수집
            results = {}
            for future in as_completed(futures):
                agent_name = futures[future]
                try:
                    result = future.result()
                    results[agent_name] = result
                except Exception as e:
                    print(f"Error in {agent_name} agent: {e}")
                    # 에러 발생 시 기본값 설정
                    if agent_name == "damage":
                        results[agent_name] = "C"  # 기본 파손 등급
                    elif agent_name == "soil":
                        results[agent_name] = "C"  # 기본 오염 등급
                    elif agent_name == "type":
                        results[agent_name] = "others"  # 기본 타입
                    elif agent_name == "material":
                        results[agent_name] = "unknown"  # 기본 재료
        
        total_time = time.time() - start_time
        
        return {
            "damage": results.get("damage", "C"),
            "soil": results.get("soil", "C"),
            "type": results.get("type", "others"),
            "material": results.get("material", "unknown"),
            "processing_time": total_time,
            "parallel_processing": True
        }

    def process_sequential(self, ref_b: bytes, used_b: bytes) -> dict:
        """
        기존 순차 처리 방식 (백업용)
        """
        start_time = time.time()
        
        # 각각의 Agent 호출 (순차 실행)
        damage_result = self.damage.analyze(ref_b, used_b)
        soil_result = self.soil.analyze(ref_b, used_b)
        type_result = self.type.analyze(used_b)
        material_result = self.material.analyze(used_b)
        
        total_time = time.time() - start_time
        
        return {
            "damage": damage_result,
            "soil": soil_result,
            "type": type_result,
            "material": material_result,
            "processing_time": total_time,
            "parallel_processing": False
        }
