import json
import concurrent.futures

from ai_agent.node_agents.type_agent import TypeAgent
from ai_agent.node_agents.material_agent import MaterialAgent
from ai_agent.node_agents.damage_agent import DamageAgent
from ai_agent.node_agents.soil_agent import SoilAgent
from ai_agent.image_input import optimize_image_size


# --- 영어 소재 → 한글 키워드로도 매핑 (기부판정 로직 호환용) ---
_EN2KO = {
    "plastic": "플라스틱",
    "metal": "금속",
    "wood": "나무",
    "fabric": "섬유",
    "silicone": "실리콘",
    "rubber": "고무",
    "paper_cardboard": "종이",
    "electronic": "전자",
    "mixed": "혼합",
    "unknown": "불명",
}

def _join_components_ko(components_en):
    ko = []
    for c in components_en or []:
        ko.append(_EN2KO.get(c, c))
    return ",".join(ko)


class SupervisorAgent:
    def __init__(self):
        self.type_agent = TypeAgent()
        self.material_agent = MaterialAgent()
        self.damage_agent = DamageAgent()
        self.soil_agent = SoilAgent()

    # 기준 이미지(ref_image_bytes)를 추가 인자로 받음 (신품 이미지)
    def process(self, front_image, left_image, rear_image, right_image, ref_image_bytes):
        import time
        start_time = time.time()
        print("🔄 이미지 최적화 중...")
        
        # 0) 각 이미지 크기 최적화
        optimized_front = optimize_image_size(front_image)
        optimized_left = optimize_image_size(left_image)
        optimized_rear = optimize_image_size(rear_image)
        optimized_right = optimize_image_size(right_image)
        ref_optimized = optimize_image_size(ref_image_bytes)

        # 1) 각 개별 에이전트를 병렬로 실행 (4개 이미지 사용)
        print("개별 에이전트 분석 중... (4개 이미지 통합 분석)")
        agent_start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_type = executor.submit(
                self.type_agent.analyze, optimized_front, optimized_left, optimized_rear, optimized_right
            )
            future_material = executor.submit(
                self.material_agent.analyze, optimized_front, optimized_left, optimized_rear, optimized_right
            )
            future_damage = executor.submit(
                self.damage_agent.analyze, optimized_front, optimized_left, optimized_rear, optimized_right, ref_optimized
            )
            future_soil = executor.submit(
                self.soil_agent.analyze, optimized_front, optimized_left, optimized_rear, optimized_right, ref_optimized
            )

            # 결과 수집 (타임아웃 15초로 단축)
            try:
                print("🚀 모든 에이전트 병렬 실행 시작...")
                
                # 모든 future를 딕셔너리로 관리
                futures = {
                    'type': future_type,
                    'material': future_material,
                    'damage': future_damage,
                    'soil': future_soil
                }
                
                # 결과를 저장할 딕셔너리
                results = {}
                
                # 완료된 순서대로 결과 수집 (진정한 병렬 처리)
                for completed_future in concurrent.futures.as_completed(futures.values(), timeout=30):
                    # 어떤 future가 완료되었는지 찾기
                    for name, future in futures.items():
                        if future == completed_future:
                            try:
                                response, tokens = completed_future.result()
                                results[name] = (response, tokens)
                                print(f"✅ {name.capitalize()}Agent 완료 (병렬)")
                                break
                            except Exception as e:
                                print(f"❌ {name.capitalize()}Agent 에러: {e}")
                                raise
                
                # 결과 언패킹
                type_response, type_tokens = results['type']
                material_response, material_tokens = results['material']
                damage_response, damage_tokens = results['damage']
                soil_response, soil_tokens = results['soil']
                
            except concurrent.futures.TimeoutError as e:
                print(f"⏰ 타임아웃 에러: {e}")
                print("일부 에이전트가 타임아웃되었습니다. 순차 처리로 전환합니다.")
                try:
                    type_response, type_tokens = self.type_agent.analyze(
                        optimized_front, optimized_left, optimized_rear, optimized_right
                    )
                    material_response, material_tokens = self.material_agent.analyze(
                        optimized_front, optimized_left, optimized_rear, optimized_right
                    )
                    damage_response, damage_tokens = self.damage_agent.analyze(
                        optimized_front, optimized_left, optimized_rear, optimized_right, ref_optimized
                    )
                    soil_response, soil_tokens = self.soil_agent.analyze(
                        optimized_front, optimized_left, optimized_rear, optimized_right, ref_optimized
                    )
                except Exception as seq_error:
                    print(f"❌ 순차 처리 중 에러: {seq_error}")
                    import traceback
                    traceback.print_exc()
                    raise seq_error
            except Exception as e:
                print(f"❌ 병렬 처리 중 에러: {e}")
                import traceback
                traceback.print_exc()
                raise e

        agent_end_time = time.time()
        agent_time = agent_end_time - agent_start_time
        print(f"🤖 에이전트 API 호출 완료: {agent_time:.2f}초")

        # 2) JSON 파싱/안전 처리
        try:
            type_result = json.loads(type_response)
        except Exception:
            type_result = {"type": "others", "battery": "불명", "size": "불명"}

        # MaterialAgent: 영어 출력 + 보조 필드 포함
        try:
            material_result = json.loads(material_response)
        except Exception:
            material_result = {
                "material": "unknown",
                "components": [],
                "secondary_hint": None,
                "confidence": 0.0,
                "material_detail": "",
                "notes": ""
            }

        # damage/soil은 dict 혹은 JSON 문자열 모두 안전 처리
        if isinstance(damage_response, dict):
            damage_result = damage_response
        else:
            try:
                damage_result = json.loads(damage_response)
            except Exception:
                damage_result = {
                    "query": {"damage": "불명", "damage_detail": "불명", "missing_parts": "불명"}
                }

        if isinstance(soil_response, dict):
            soil_result = soil_response
        else:
            try:
                soil_result = json.loads(soil_response)
            except Exception:
                soil_result = {"query": {"soil": "깨끗", "soil_detail": "오염 없음"}}

        # 3) 통합 판단 로직 적용
        # type: 영어 카테고리를 그대로 사용 (최종 출력도 영어)
        toy_type = type_result.get("type", "others")
        battery = type_result.get("battery", "불명")
        size = type_result.get("size", "중간")

        # material: 영어 그대로 출력하되, 기부판정 로직 호환 위해 한글 문자열도 생성
        material_en = (material_result.get("material") or "unknown").strip().lower()
        components_en = material_result.get("components") or []
        secondary_hint = material_result.get("secondary_hint")
        material_detail = material_result.get("material_detail", "")
        material_notes = material_result.get("notes", "")
        confidence = material_result.get("confidence", 0.0)

        # 최종 노출용(영어)
        if material_en == "mixed" and components_en:
            material_display = "mixed(" + ",".join(components_en) + ")"
        else:
            material_display = material_en

        # 기부판정 로직용(한글 조합)
        if material_en == "mixed" and components_en:
            material_for_rules_ko = _join_components_ko(components_en)  # 예: "플라스틱,금속"
        else:
            material_for_rules_ko = _EN2KO.get(material_en, material_en)

        if material_notes:
            material_detail = (material_detail + " | " + material_notes).strip(" |")

        # damage는 query 기준 라벨들 읽기
        if "query" in damage_result:
            damage = damage_result["query"].get("damage", "불명")
            damage_detail = damage_result["query"].get("damage_detail", "불명")
            missing_parts = damage_result["query"].get("missing_parts", "불명")
        else:
            damage = damage_result.get("damage", "불명")
            damage_detail = damage_result.get("damage_detail", "불명")
            missing_parts = damage_result.get("missing_parts", "불명")

        # soil은 query 기준 라벨들 읽기
        if "query" in soil_result:
            soil = soil_result["query"].get("soil", "깨끗")
            soil_detail = soil_result["query"].get("soil_detail", "오염 없음")
        else:
            soil = soil_result.get("soil", "깨끗")
            soil_detail = soil_result.get("soil_detail", "오염 없음")

        # --- Damage: 상대 등급/점수 (신규 포맷) ---
        rel_damage_grade = None
        query_damage_score = None
        reference_damage_score = None
        if isinstance(damage_result, dict):
            # 다양한 키 호환
            rel_damage_grade = (
                damage_result.get("relative_damage_grade")
                or damage_result.get("grade")
                or (damage_result.get("relative_vs_ref") or {}).get("grade")
            )
            query_damage_score = (
                damage_result.get("query_damage_score")
                or damage_result.get("query_abs")
            )
            reference_damage_score = (
                damage_result.get("reference_damage_score")
                or damage_result.get("reference_abs")
            )

        # --- Soil: 등급/수치 추출 (여러 포맷 호환) ---
        soil_grade = None
        soil_level = None
        if isinstance(soil_result, dict):
            # 등급 우선: 최신 포맷 우선순위대로 조회
            soil_grade = (
                soil_result.get("relative_soil_grade")
                or soil_result.get("grade")
                or (soil_result.get("relative_vs_ref") or {}).get("grade")
            )

            # 수치(-2..+2 혹은 delta): 있으면 함께 전달(디버그/호환용)
            lvl = soil_result.get("soil_level")
            if lvl is None:
                lvl = (soil_result.get("relative_vs_ref") or {}).get("delta")
            soil_level = lvl

        # 의미있는 관찰사항 생성
        notes = self._generate_meaningful_notes(
            toy_type, battery, material_display, material_detail,
            damage, damage_detail, missing_parts, soil, soil_detail
        )

        # 4) 기부 판단 로직 (한글 호환 문자열로 평가)
        donate, donate_reason, repair_or_disassemble = self._judge_donation(
            toy_type, battery, material_for_rules_ko, material_detail,
            damage, damage_detail, missing_parts, soil, soil_detail
        )

        # 5) 토큰 사용량 합산
        token_usage = {
            "type_agent": type_tokens.get("total_tokens", 0),
            "material_agent": material_tokens.get("total_tokens", 0),
            "damage_agent": damage_tokens.get("total_tokens", 0),
            "soil_agent": soil_tokens.get("total_tokens", 0),
            "total": type_tokens.get("total_tokens", 0)
                     + material_tokens.get("total_tokens", 0)
                     + damage_tokens.get("total_tokens", 0)
                     + soil_tokens.get("total_tokens", 0),
        }

        # 6) 결과 반환 (영어 출력 유지)
        end_time = time.time()
        total_time = end_time - start_time
        print(f"⚡ 에이전트 분석 완료: {total_time:.2f}초")
        
        return {
            "장난감 종류": toy_type,                    # 영어 카테고리
            "건전지 여부": battery,
            "재료": material_display,                  # 영어 (mixed는 mixed(plastic,metal) 형태)
            "파손": damage,
            "오염도": soil,
            "관찰사항": notes,
            "크기": size,

            "기부 가능 여부": "가능" if donate else "불가능",
            "기부 불가 사유": donate_reason,
            "수리/분해": repair_or_disassemble,

            "토큰 사용량": token_usage,

            # DamageAgent
            "파손 상대등급": rel_damage_grade,                 # 'A'..'E'
            "파손 점수(중고)": query_damage_score,            # 0..4
            "파손 점수(신품)": reference_damage_score,        # 0..4

            # SoilAgent (신규/구버전 혼용 지원)
            "오염 상대등급": soil_grade,                      # 'A'..'E' (있다면)
            "기준 대비(오염)": int(soil_level) if soil_level is not None else None,
        }

    # --- 이하 기존 judge/notes 메서드는 그대로 (한국어 키워드 기반) ---
    def _judge_donation(self, toy_type, battery, material, material_detail, damage, damage_detail, missing_parts, soil, soil_detail):
        """
        가중치 기반 기부 가능 여부 판단 시스템
        소재(40%) + 부품상태(30%) + 파손(20%) + 오염도(10%) 순으로 평가
        """
        if material == "나무":
            return False, "나무 소재는 안전상 기부 불가", "분해/부품 추출(업사이클)"
        if ("천" in material or "섬유" in material or 
            material == "섬유" or "섬유" in material_detail or
            "천" in str(material_detail) or "섬유" in str(material_detail)):
            return False, "천/섬유 소재는 위생상 기부 불가", "분해/부품 추출(업사이클)"
        if ("실리콘" in material or "실리콘" in str(material_detail) or
            "고무" in material or "고무" in str(material_detail)):
            return False, "실리콘/고무 소재는 재활용 불가능", "분해/부품 추출(업사이클)"
        if (("천" in material or "섬유" in material) and 
            ("," in material or "혼합" in material_detail or "혼합" in material)):
            return False, "천/섬유가 포함된 혼합 소재는 위생상 기부 불가", "분해/부품 추출(업사이클)"
        if (("실리콘" in material or "고무" in material) and 
            ("," in material or "혼합" in material_detail or "혼합" in material)):
            return False, "실리콘/고무가 포함된 혼합 소재는 재활용 불가능", "분해/부품 추출(업사이클)"
        if "심각" in damage or damage == "심각한 파손" or "심각" in damage_detail:
            return False, "심각한 파손으로 완제품 기부 불가", "분해/부품 추출(업사이클)"
        if soil == "더러움" or "더러움" in soil or "재활용 불가" in str(soil_detail):
            return False, "오염 상태로 위생상 기부 불가", "분해/부품 추출(업사이클)"
        if toy_type in ["인형", "아동 도서", "보행기", "탈것"]:
            return False, f"{toy_type}은 기부 불가 종류", "분해/부품 추출(업사이클)"
        if ("부품" in str(toy_type) or "용도 불분명" in str(toy_type) or 
            "불분명" in str(toy_type)):
            return False, f"{toy_type}은 기부 불가", "분해/부품 추출(업사이클)"
        score = 0
        if material == "플라스틱" and "단일" in material_detail:
            score += 40
        elif material == "금속" and "단일" in material_detail:
            score += 35
        elif "플라스틱,금속" in material or ("플라스틱" in material and "금속" in material):
            score += 35
        elif "플라스틱,천" in material or "플라스틱,섬유" in material or ("플라스틱" in material and ("천" in material or "섬유" in material)):
            score += 0
        elif "플라스틱,실리콘" in material or ("플라스틱" in material and "실리콘" in material):
            score += 25
        elif "혼합" in material_detail or "," in material:
            score += 20
        else:
            score += 20
        if missing_parts == "있음" or "부품" in damage or "일부" in damage:
            score += 0
        elif missing_parts == "불명" or "판단 불가" in damage:
            score += 5
        elif missing_parts == "없음" and damage == "없음":
            score += 40
        elif missing_parts == "없음" and ("미세" in damage or "경미" in damage):
            score += 25
        else:
            score += 15
        if damage == "없음":
            score += 20
        elif "미세" in damage or "경미" in damage or "미세" in damage_detail or "경미" in damage_detail:
            score += 15
        elif "파손" in damage or "부서" in damage or "파손" in damage_detail or "부서" in damage_detail:
            score += 5
        else:
            score += 10
        if soil == "깨끗" or "깨끗" in soil or "깨끗" in soil_detail:
            score += 10
        elif soil == "보통" or "약간 더러움" in soil or "보통" in soil_detail:
            score += 5
        else:
            score += 0
        if "용도 불분명" in str(toy_type) or "불분명" in str(toy_type):
            score -= 30
        if "부품" in str(toy_type) or "완제품" in str(toy_type):
            score -= 25
        if battery == "건전지" and (missing_parts == "있음" or "부품" in damage):
            score -= 20
        if score >= 75:
            if battery == "건전지":
                return True, "플라스틱 건전지 장난감, 상태 양호", "수리 불필요(완제품)"
            else:
                return True, "플라스틱 비건전지 장난감, 상태 양호", "수리 불필요(완제품)"
        elif score >= 55:
            return False, "경미한 문제로 수리 필요", "경미 수리 권장"
        elif score >= 35:
            return False, "여러 문제로 기부 어려움", "분해/부품 추출(업사이클)"
        else:
            return False, "심각한 문제로 기부 불가", "분해/부품 추출(업사이클)"

    def _generate_meaningful_notes(self, toy_type, battery, material, material_detail, damage, damage_detail, missing_parts, soil, soil_detail):
        notes = []
        if toy_type == "action_figures" or toy_type == "others":
            notes.append("피규어/모형류는 세밀한 파손 여부 확인 필요")
        elif toy_type == "vehicles":
            notes.append("자동차 장난감은 바퀴와 동작부위 상태 중요")
        elif toy_type == "robot":
            notes.append("변신 로봇은 관절부위 파손 여부 확인 필요")
        elif toy_type == "building_blocks":
            notes.append("블록류는 결합부위 마모 상태 확인")
        elif toy_type == "sports":
            notes.append("스포츠 장난감은 마모/변형 확인")

        if battery == "건전지":
            notes.append("건전지 장난감은 전자부품 상태 확인 필요")
        elif battery == "비건전지":
            notes.append("비건전지 장난감은 기계적 동작 확인")

        # 소재 관련 안내 (영문 출력값을 그대로 주석처럼 사용)
        if "plastic" in material:
            notes.append("플라스틱 소재는 균열이나 변형 확인")
        elif "metal" in material:
            notes.append("금속 소재는 녹이나 변형 상태 확인")
        elif "wood" in material:
            notes.append("나무 소재는 균열이나 부식 상태 확인")
        elif "fabric" in material:
            notes.append("섬유/천 소재는 위생상태와 마모 확인")
        elif "silicone" in material or "rubber" in material:
            notes.append("실리콘/고무 소재는 표면 오염 및 경화 확인")

        if damage == "없음" and missing_parts == "없음":
            notes.append("파손 없음, 부품 완전 - 기부 적합")
        elif missing_parts == "있음":
            notes.append("부품 누락 - 부품 보완 후 기부 가능")
        elif missing_parts == "불명":
            notes.append("부품 상태 불명 - 추가 확인 필요")
        elif "미세" in damage or "경미" in damage or "미세" in damage_detail or "경미" in damage_detail:
            notes.append("경미한 파손 - 수리 후 기부 가능")
        elif "심각" in damage or "심각" in damage_detail:
            notes.append("심각한 파손 - 부품 추출 후 업사이클")

        if soil == "깨끗" or "깨끗" in soil or "깨끗" in soil_detail:
            notes.append("오염 없음 - 기부 적합")
        elif soil == "보통" or "보통" in soil_detail:
            notes.append("약간의 사용 흔적 - 세척 후 기부 가능")
        elif soil == "약간 더러움" or "약간 더러움" in soil_detail:
            notes.append("약간 더러움 - 세척 후 기부 가능")
        elif soil == "더러움" or "더러움" in soil or "더러움" in soil_detail:
            notes.append("심한 오염 - 위생상 기부 불가")

        return " | ".join(notes) if notes else "기본적인 상태 확인 완료"
