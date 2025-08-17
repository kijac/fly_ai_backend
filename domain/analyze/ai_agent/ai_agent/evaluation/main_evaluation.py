#!/usr/bin/env python3
"""
수정된 에이전트들을 사용해서 CSV 파일을 기반으로 AI 에이전트의 기부 가능여부 판별 정확도를 평가하는 스크립트
"""

import json
import os
import pandas as pd
import sys
from datetime import datetime
from pathlib import Path

# 상위 디렉토리를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent.parent))

from ai_agent.supervisor_agent import SupervisorAgent

class ImprovedCSVDonationEvaluator:
    def __init__(self):
        self.supervisor = SupervisorAgent()
        self.results_dir = Path("evaluation_results")
        self.results_dir.mkdir(exist_ok=True)
    
    def evaluate_from_csv(self, csv_path):
        """CSV 파일을 읽어서 AI 에이전트의 성능을 평가합니다."""
        print("🎯 수정된 에이전트 기반 기부 가능 여부 평가 시작...")
        
        # CSV 파일 로드
        try:
            df = pd.read_csv(csv_path, header=1)
            print(f"📁 CSV 파일 로드 완료: {len(df)}개 행")
        except Exception as e:
            print(f"❌ CSV 파일 로드 실패: {e}")
            return None, 0
        
        # 컬럼명 확인
        print(f"📊 컬럼명: {list(df.columns)}")
        
        # 필요한 컬럼 확인
        required_columns = ['IMG_NO', '이미지 (전, 좌, 후, 우)', '기부 가능 여부 (리사이클링)']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"❌ 필요한 컬럼이 없습니다: {missing_columns}")
            return None, 0
        
        print("✅ 필요한 컬럼 확인 완료")
        
        # 실제 존재하는 이미지 파일 목록 생성
        images_dir = Path("datasets/images")
        existing_images = set()
        if images_dir.exists():
            for img_file in images_dir.glob("img_*_front.*"):
                img_no = img_file.stem.split('_')[1]  # img_0000_front -> 0000
                existing_images.add(img_no)
        
        print(f"📸 실제 존재하는 이미지: {len(existing_images)}개 (img_0000 ~ img_{max(existing_images) if existing_images else '0000'})")
        
        results = []
        total_tokens = 0
        processed_count = 0
        
        # 각 행 처리 (처음 100개만 테스트)
        max_test_count = 100
        for idx, row in df.iterrows():
            if processed_count >= max_test_count:
                print(f"   📊 테스트 제한에 도달했습니다. ({max_test_count}개)")
                break
                
            try:
                # 이미지 번호 추출
                img_no = str(row['IMG_NO']).zfill(4)  # 0 -> 0000
                
                # 실제 존재하는 이미지만 처리
                if img_no not in existing_images:
                    continue
                
                # 이미지 파일 경로 구성 (4개 이미지 사용)
                front_img = f"img_{img_no}_front"
                left_img = f"img_{img_no}_left"
                rear_img = f"img_{img_no}_rear"
                right_img = f"img_{img_no}_right"
                
                image_paths = {
                    'front': images_dir / f"{front_img}.png",
                    'left': images_dir / f"{left_img}.png",
                    'rear': images_dir / f"{rear_img}.png",
                    'right': images_dir / f"{right_img}.png"
                }
                
                # 이미지 파일 존재 확인 (4개 이미지)
                if not all(path.exists() for path in image_paths.values()):
                    print(f"⚠️ 4개 이미지를 모두 찾을 수 없습니다: {front_img}")
                    continue
                
                print(f"🔍 [{processed_count+1}] {front_img} (4개 각도) 분석 중...")
                
                # 4개 이미지 파일 읽기
                try:
                    front_image_bytes = open(image_paths['front'], 'rb').read()
                    left_image_bytes = open(image_paths['left'], 'rb').read()
                    rear_image_bytes = open(image_paths['rear'], 'rb').read()
                    right_image_bytes = open(image_paths['right'], 'rb').read()
                except Exception as e:
                    print(f"❌ 이미지 파일 읽기 실패: {e}")
                    continue
                
                # AI 에이전트로 분석 (4개 이미지 전달)
                try:
                    ai_result = self.supervisor.process(front_image_bytes, left_image_bytes, rear_image_bytes, right_image_bytes)
                    
                    # 실제 기부 가능 여부 (ground truth)
                    ground_truth = row['기부 가능 여부 (리사이클링)']
                    if pd.isna(ground_truth):
                        ground_truth = "불명"
                    
                    # AI 예측 결과
                    ai_prediction = ai_result['기부 가능 여부']
                    
                    # 각 agent별 결과 추출
                    damage_result = ai_result.get('파손', '불명')
                    material_result = ai_result.get('재료', '불명')
                    soil_result = ai_result.get('오염도', '불명')
                    type_result = ai_result.get('장난감 종류', '불명')
                    
                    # Ground truth 결과 추출
                    ground_truth_damage = row.get('부품이 모두 있는가', '불명')
                    ground_truth_material = row.get('소재', '불명')
                    ground_truth_soil = row.get('오염도', '불명')
                    ground_truth_type = row.get('장난감 종류', '불명')
                    
                    # 정확도 계산
                    is_correct = self._calculate_accuracy(ai_prediction, ground_truth)
                    
                    # DamageAgent는 부품 완전성과 파손 여부가 반대 관계
                    damage_ai_prediction = damage_result
                    damage_ground_truth = ground_truth_damage
                    
                    # AI가 "없음"이면 실제로는 "있음"이어야 정확
                    if damage_ai_prediction == "없음" and damage_ground_truth == "있음":
                        damage_correct = True
                    elif damage_ai_prediction == "있음" and damage_ground_truth == "없음":
                        damage_correct = True
                    elif damage_ai_prediction == "불명" and damage_ground_truth == "불명":
                        damage_correct = True
                    else:
                        damage_correct = False
                    
                    material_correct = self._calculate_accuracy(material_result, ground_truth_material)
                    soil_correct = self._calculate_accuracy(soil_result, ground_truth_soil)
                    type_correct = self._calculate_accuracy(type_result, ground_truth_type)
                    
                    # 결과 저장
                    result = {
                        'img_no': img_no,
                        'image_file': front_img,
                        'ground_truth': ground_truth,
                        'ai_prediction': ai_prediction,
                        'is_correct': is_correct,
                        'ai_details': ai_result,
                        'agent_results': {
                            'damage': {'predicted': damage_result, 'actual': ground_truth_damage, 'correct': damage_correct},
                            'material': {'predicted': material_result, 'actual': ground_truth_material, 'correct': material_correct},
                            'soil': {'predicted': soil_result, 'actual': ground_truth_soil, 'correct': soil_correct},
                            'type': {'predicted': type_result, 'actual': ground_truth_type, 'correct': type_correct}
                        },
                        'ground_truth_details': {
                            'toy_type': row.get('장난감 종류', '불명'),
                            'material': row.get('소재', '불명'),
                            'parts_complete': row.get('부품이 모두 있는가', '불명'),
                            'soil_level': row.get('오염도', '불명'),
                            'recycling_reason': row.get('리사이클링 또는 업사이클링 불가 이유', '')
                        }
                    }
                    
                    results.append(result)
                    
                    # 토큰 사용량 누적
                    if '토큰 사용량' in ai_result:
                        total_tokens += ai_result['토큰 사용량'].get('total', 0)
                    
                    processed_count += 1
                    
                    # 진행상황 출력
                    status = "✅" if is_correct else "❌"
                    print(f"   {status} 예측: {ai_prediction} | 실제: {ground_truth}")
                    
                except Exception as e:
                    print(f"❌ AI 분석 실패: {e}")
                    continue
                
            except Exception as e:
                print(f"❌ 행 처리 실패 (행 {idx}): {e}")
                continue
        
        # 전체 정확도 계산
        if results:
            correct_count = sum(1 for r in results if r['is_correct'])
            overall_accuracy = (correct_count / len(results)) * 100
            
            # 각 agent별 정확도 계산
            damage_correct_count = sum(1 for r in results if r['agent_results']['damage']['correct'])
            material_correct_count = sum(1 for r in results if r['agent_results']['material']['correct'])
            soil_correct_count = sum(1 for r in results if r['agent_results']['soil']['correct'])
            type_correct_count = sum(1 for r in results if r['agent_results']['type']['correct'])
            
            damage_accuracy = (damage_correct_count / len(results)) * 100
            material_accuracy = (material_correct_count / len(results)) * 100
            soil_accuracy = (soil_correct_count / len(results)) * 100
            type_accuracy = (type_correct_count / len(results)) * 100
            
            print(f"\n📊 평가 완료!")
            print(f"총 처리된 이미지: {len(results)}개")
            print(f"전체 기부 가능 여부 정확도: {overall_accuracy:.2f}% ({correct_count}/{len(results)})")
            print(f"\n🔍 각 Agent별 정확도:")
            print(f"  DamageAgent (파손): {damage_accuracy:.2f}% ({damage_correct_count}/{len(results)})")
            print(f"  MaterialAgent (재료): {material_accuracy:.2f}% ({material_correct_count}/{len(results)})")
            print(f"  SoilAgent (오염도): {soil_accuracy:.2f}% ({soil_correct_count}/{len(results)})")
            print(f"  TypeAgent (종류): {type_accuracy:.2f}% ({type_correct_count}/{len(results)})")
            print(f"\n💰 총 토큰 사용량: {total_tokens:,}개")
            
            # 결과 저장
            self._save_results(results, overall_accuracy, csv_path)
            
            # 상세 분석 출력
            self._print_detailed_analysis(results)
            
            # 10개 결과를 xlsx와 csv로 정리
            self._export_10_results_to_files(results)
            
            # MaterialAgent 상세 분석 (일단 주석 처리)
            # self._print_material_analysis(results)
            
            # 모든 Agent 상세 분석 (Excel/CSV 포함) (일단 주석 처리)
            # self._print_all_agents_analysis(results)
            
            return results, overall_accuracy
        else:
            print("❌ 평가할 수 있는 결과가 없습니다.")
            return None, 0
    
    def _save_results(self, results, overall_accuracy, csv_path):
        """평가 결과를 파일로 저장합니다."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = Path(csv_path).stem
        
        # JSON 결과 저장
        json_filename = f"improved_evaluation_{csv_filename}_{timestamp}.json"
        json_path = self.results_dir / json_filename
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'evaluation_summary': {
                    'timestamp': timestamp,
                    'total_images': len(results),
                    'overall_accuracy': overall_accuracy,
                    'csv_source': csv_path
                },
                'detailed_results': results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"💾 JSON 결과 저장 완료: {json_path}")
        
        # CSV 요약 저장
        csv_summary_filename = f"improved_evaluation_summary_{csv_filename}_{timestamp}.csv"
        csv_summary_path = self.results_dir / csv_summary_filename
        
        summary_data = []
        for result in results:
            summary_data.append({
                'IMG_NO': result['img_no'],
                '이미지파일': result['image_file'],
                '실제_기부가능여부': result['ground_truth'],
                'AI_예측_기부가능여부': result['ai_prediction'],
                '정확도': '정확' if result['is_correct'] else '오류',
                'AI_장난감종류': result['ai_details'].get('장난감 종류', ''),
                'AI_재료': result['ai_details'].get('재료', ''),
                'AI_파손': result['ai_details'].get('파손', ''),
                'AI_오염도': result['ai_details'].get('오염도', ''),
                'AI_기부불가사유': result['ai_details'].get('기부 불가 사유', ''),
                '실제_장난감종류': result['ground_truth_details']['toy_type'],
                '실제_재료': result['ground_truth_details']['material'],
                '실제_부품완전성': result['ground_truth_details']['parts_complete'],
                '실제_오염도': result['ground_truth_details']['soil_level']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(csv_summary_path, index=False, encoding='utf-8-sig')
        
        print(f"💾 CSV 요약 저장 완료: {csv_summary_path}")
    
    def _print_detailed_analysis(self, results):
        """상세한 분석 결과를 출력합니다."""
        print(f"\n🔍 상세 분석 결과:")
        
        # 에이전트별 정확도 분석
        correct_results = [r for r in results if r['is_correct']]
        error_results = [r for r in results if not r['is_correct']]
        
        print(f"✅ 정확한 예측 ({len(correct_results)}개):")
        for result in correct_results[:5]:  # 처음 5개만 출력
            print(f"   - {result['image_file']}: {result['ai_prediction']}")
        
        if len(correct_results) > 5:
            print(f"   ... 및 {len(correct_results) - 5}개 더")
        
        print(f"\n❌ 오류 예측 ({len(error_results)}개):")
        for result in error_results[:5]:  # 처음 5개만 출력
            print(f"   - {result['image_file']}: AI={result['ai_prediction']}, 실제={result['ground_truth']}")
            if '기부 불가 사유' in result['ai_details']:
                print(f"     AI 판단 사유: {result['ai_details']['기부 불가 사유']}")
        
        if len(error_results) > 5:
            print(f"   ... 및 {len(error_results) - 5}개 더")

    def _print_material_analysis(self, results):
        """MaterialAgent 결과를 상세하게 분석하고 CSV로 출력합니다."""
        print("\n🔍 MaterialAgent 상세 분석 결과:")
        
        # MaterialAgent 결과만 추출
        material_results = []
        for result in results:
            material_info = result['agent_results']['material']
            ground_truth_info = result['ground_truth_details']
            
            material_results.append({
                'img_no': result['img_no'],
                'image_file': result['image_file'],
                'ai_predicted_material': material_info['predicted'],
                'actual_material': material_info['actual'],
                'is_correct': material_info['correct'],
                'ai_confidence': self._extract_ai_confidence(result['ai_details'], '재료'),
                'ai_notes': self._extract_ai_notes(result['ai_details'], '재료')
            })
        
        # DataFrame 생성
        import pandas as pd
        df_material = pd.DataFrame(material_results)
        
        # 정확도 통계
        correct_count = df_material['is_correct'].sum()
        total_count = len(df_material)
        accuracy = (correct_count / total_count) * 100
        
        print(f"📊 MaterialAgent 정확도: {accuracy:.2f}% ({correct_count}/{total_count})")
        
        # 오답 분석
        wrong_predictions = df_material[~df_material['is_correct']]
        if len(wrong_predictions) > 0:
            print(f"\n❌ 오답 케이스 ({len(wrong_predictions)}개):")
            for _, row in wrong_predictions.iterrows():
                print(f"  {row['img_no']}: AI={row['ai_predicted_material']} | 실제={row['actual_material']}")
        
        # CSV 저장
        csv_filename = f"material_agent_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        csv_path = Path("evaluation_results") / csv_filename
        df_material.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 MaterialAgent 분석 결과 저장: {csv_path}")
        
        # 상세 결과 출력 (처음 10개)
        print(f"\n📋 상세 결과 (처음 10개):")
        print(df_material.head(10).to_string(index=False))
        
        return df_material
    
    def _print_all_agents_analysis(self, results):
        """모든 Agent의 결과를 상세하게 분석하고 Excel/CSV로 출력합니다."""
        print("\n🔍 모든 Agent 상세 분석 결과:")
        
        # 모든 Agent 결과 추출
        all_agents_results = []
        for result in results:
            # 기본 정보
            base_info = {
                'img_no': result['img_no'],
                'image_file': result['image_file'],
                'overall_prediction': result['ai_prediction'],
                'overall_actual': result['ground_truth'],
                'overall_correct': result['is_correct']
            }
            
            # 각 Agent별 결과
            agents_info = {}
            for agent_name, agent_result in result['agent_results'].items():
                agents_info.update({
                    f'{agent_name}_predicted': agent_result['predicted'],
                    f'{agent_name}_actual': agent_result['actual'],
                    f'{agent_name}_correct': agent_result['correct']
                })
            
            # AI 상세 정보
            ai_details = result['ai_details']
            ai_info = {}
            for agent_name in ['파손', '재료', '오염도', '장난감 종류']:
                if agent_name in ai_details:
                    agent_data = ai_details[agent_name]
                    if isinstance(agent_data, dict):
                        ai_info[f'{agent_name}_confidence'] = agent_data.get('confidence', '불명')
                        ai_info[f'{agent_name}_notes'] = str(agent_data.get('notes', ''))[:100]  # 100자 제한
                    else:
                        ai_info[f'{agent_name}_confidence'] = '불명'
                        ai_info[f'{agent_name}_notes'] = str(agent_data)[:100]
                else:
                    ai_info[f'{agent_name}_confidence'] = '불명'
                    ai_info[f'{agent_name}_notes'] = ''
            
            # Ground Truth 상세 정보
            gt_info = {
                'gt_toy_type': result['ground_truth_details']['toy_type'],
                'gt_material': result['ground_truth_details']['material'],
                'gt_parts_complete': result['ground_truth_details']['parts_complete'],
                'gt_soil_level': result['ground_truth_details']['soil_level'],
                'gt_recycling_reason': result['ground_truth_details']['recycling_reason'][:100]  # 100자 제한
            }
            
            # 모든 정보 합치기
            row_data = {**base_info, **agents_info, **ai_info, **gt_info}
            all_agents_results.append(row_data)
        
        # DataFrame 생성
        import pandas as pd
        df_all = pd.DataFrame(all_agents_results)
        
        # 각 Agent별 정확도 계산
        agent_accuracies = {}
        for agent_name in ['damage', 'material', 'soil', 'type']:
            correct_col = f'{agent_name}_correct'
            if correct_col in df_all.columns:
                correct_count = df_all[correct_col].sum()
                total_count = len(df_all)
                accuracy = (correct_count / total_count) * 100
                agent_accuracies[agent_name] = accuracy
                print(f"📊 {agent_name.capitalize()}Agent 정확도: {accuracy:.2f}% ({correct_count}/{total_count})")
        
        # 전체 정확도
        overall_correct = df_all['overall_correct'].sum()
        overall_total = len(df_all)
        overall_accuracy = (overall_correct / overall_total) * 100
        print(f"📊 전체 기부 가능 여부 정확도: {overall_accuracy:.2f}% ({overall_correct}/{overall_total})")
        
        # Excel 저장
        excel_filename = f"all_agents_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        excel_path = Path("evaluation_results") / excel_filename
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # 전체 결과
            df_all.to_excel(writer, sheet_name='전체결과', index=False)
            
            # 각 Agent별 결과
            for agent_name in ['damage', 'material', 'soil', 'type']:
                agent_df = df_all[['img_no', 'image_file', 
                                  f'{agent_name}_predicted', f'{agent_name}_actual', 
                                  f'{agent_name}_correct',
                                  f'{agent_name}_confidence', f'{agent_name}_notes']].copy()
                agent_df.columns = ['이미지번호', '이미지파일', 'AI예측', '실제정답', '정확여부', 'AI신뢰도', 'AI설명']
                agent_df.to_excel(writer, sheet_name=f'{agent_name.capitalize()}Agent', index=False)
            
            # 오답 분석
            wrong_df = df_all[~df_all['overall_correct']].copy()
            if len(wrong_df) > 0:
                wrong_df.to_excel(writer, sheet_name='오답분석', index=False)
        
        print(f"\n💾 Excel 파일 저장 완료: {excel_path}")
        
        # CSV 저장
        csv_filename = f"all_agents_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        csv_path = Path("evaluation_results") / csv_filename
        df_all.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"💾 CSV 파일 저장 완료: {csv_path}")
        
        # 상세 결과 미리보기 (처음 5개)
        print(f"\n📋 상세 결과 미리보기 (처음 5개):")
        preview_columns = ['img_no', 'image_file', 'overall_prediction', 'overall_actual', 
                          'damage_predicted', 'damage_actual', 'material_predicted', 'material_actual',
                          'soil_predicted', 'soil_actual', 'type_predicted', 'type_actual']
        available_columns = [col for col in preview_columns if col in df_all.columns]
        print(df_all[available_columns].head().to_string(index=False))
        
        return df_all
    
    def _export_10_results_to_files(self, results):
        """10개 결과를 xlsx와 csv로 정리하여 저장합니다."""
        if not results:
            print("❌ 결과가 없어 파일로 내보낼 수 없습니다.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"evaluation_results_{timestamp}.csv"
        xlsx_filename = f"evaluation_results_{timestamp}.xlsx"

        csv_path = self.results_dir / csv_filename
        xlsx_path = self.results_dir / xlsx_filename

        # CSV 요약 데이터 추출
        summary_data = []
        for result in results:
            summary_data.append({
                'IMG_NO': result['img_no'],
                '이미지파일': result['image_file'],
                '실제_기부가능여부': result['ground_truth'],
                'AI_예측_기부가능여부': result['ai_prediction'],
                '정확도': '정확' if result['is_correct'] else '오류',
                'AI_장난감종류': result['ai_details'].get('장난감 종류', ''),
                'AI_재료': result['ai_details'].get('재료', ''),
                'AI_파손': result['ai_details'].get('파손', ''),
                'AI_오염도': result['ai_details'].get('오염도', ''),
                'AI_기부불가사유': result['ai_details'].get('기부 불가 사유', ''),
                '실제_장난감종류': result['ground_truth_details']['toy_type'],
                '실제_재료': result['ground_truth_details']['material'],
                '실제_부품완전성': result['ground_truth_details']['parts_complete'],
                '실제_오염도': result['ground_truth_details']['soil_level']
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"💾 10개 결과 요약 CSV 저장: {csv_path}")

        # 상세 결과 데이터 추출
        detailed_data = []
        for result in results:
            detailed_data.append({
                '이미지번호': result['img_no'],
                '이미지파일': result['image_file'],
                '실제_기부가능여부': result['ground_truth'],
                'AI_예측_기부가능여부': result['ai_prediction'],
                '정확여부': '정확' if result['is_correct'] else '오류',
                'AI_장난감종류': result['ai_details'].get('장난감 종류', ''),
                'AI_재료': result['ai_details'].get('재료', ''),
                'AI_파손': result['ai_details'].get('파손', ''),
                'AI_오염도': result['ai_details'].get('오염도', ''),
                'AI_기부불가사유': result['ai_details'].get('기부 불가 사유', ''),
                '실제_장난감종류': result['ground_truth_details']['toy_type'],
                '실제_재료': result['ground_truth_details']['material'],
                '실제_부품완전성': result['ground_truth_details']['parts_complete'],
                '실제_오염도': result['ground_truth_details']['soil_level']
            })

        detailed_df = pd.DataFrame(detailed_data)
        detailed_df.to_csv(csv_path, mode='a', header=False, encoding='utf-8-sig') # 헤더 없이 추가
        print(f"💾 10개 결과 상세 CSV 저장: {csv_path}")

        # Excel 저장
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            # 전체 결과
            summary_df.to_excel(writer, sheet_name='요약', index=False)
            detailed_df.to_excel(writer, sheet_name='상세', index=False)
        
        print(f"💾 10개 결과 Excel 저장: {xlsx_path}")

    def _normalize_text(self, text):
        """텍스트를 정규화하여 매칭을 개선합니다."""
        if pd.isna(text) or text == '':
            return '불명'
        
        text = str(text).strip().lower()
        
        # 숫자로 된 텍스트 처리
        if text.isdigit():
            return text
        
        # 유사한 의미의 텍스트 매핑
        text_mapping = {
            # 파손/부품 관련 - DamageAgent 정확도 향상
            '있음': ['있음', '있다', '파손', '손상', '부서짐', '깨짐', '찢어짐', '파손됨', '손상됨', '있습니다', '있음'],
            '없음': ['없음', '없다', '파손없음', '손상없음', '완전함', '양호', '좋음', '완벽한 상태', '완벽한상태', '없습니다', '없음'],
            '불명': ['불명', '모름', '확실하지않음', '판단불가', '확실하지않음', '확실하지 않음', '판단 불가'],
            '경미한 파손': ['경미한 파손', '경미한파손', '미세한 파손', '미세한파손', '약간의 파손', '약간의파손'],
            '심각한 파손': ['심각한 파손', '심각한파손', '큰 파손', '큰파손', '심한 파손', '심한파손'],
            
            # 재료 관련 - 정확한 매칭을 위해 부분 매칭 제거
            '플라스틱': ['플라스틱'],
            '플라스틱, 천': ['플라스틱, 천', '천, 플라스틱'],
            '플라스틱, 금속': ['플라스틱, 금속', '금속, 플라스틱'],
            '플라스틱, 고무': ['플라스틱, 고무', '고무, 플라스틱'],
            '플라스틱, 섬유': ['플라스틱, 섬유', '섬유, 플라스틱'],
            '금속': ['금속', 'metal', '철', '강철', '알루미늄'],
            '섬유': ['섬유', '천', '면', '실', 'fabric', 'cloth'],
            '고무': ['고무', 'rubber', '실리콘'],
            '나무': ['나무', 'wood', '목재'],
            '실리콘': ['실리콘', 'silicone'],
            
            # 오염도 관련 - SoilAgent 정확도 향상
            '깨끗': ['깨끗', '깨끗함', '오염없음', '청결', '깨끗 (미세소독 필요)', '깨끗(미세소독 필요)'],
            '보통': ['보통', '보통수준', '약간더러움', '약간 더러움', '사용흔적', '약간의 사용흔적'],
            '더러움': ['더러움', '더럽다', '심한오염', '심한 오염', '오염', '매우더러움'],
            
            # 장난감 종류 관련 - TypeAgent 정확도 향상
            '피규어': ['피규어', 'figure', '인형'],
            '자동차': ['자동차', 'car', '차', '탈것', '자동차 장난감'],
            '변신로봇': ['변신로봇', '로봇', 'robot', '변신 로봇'],
            '건전지장난감': ['건전지장난감', '건전지 장난감', '전자장난감', '건전지 장난감 (사운드북 포함)'],
            '비건전지장난감': ['비건전지장난감', '비건전지 장난감', '기계식장난감'],
            '블록': ['블록', 'block', '레고'],
            '공': ['공', 'ball', '구'],
            '기타': ['기타', 'other', '기타장난감']
        }
        
        # 정확한 매칭 시도
        for key, values in text_mapping.items():
            if text in values:
                return key
        
        # 부분 매칭이 아닌 정확한 매칭만 허용
        return text
    
    def _calculate_accuracy(self, predicted, actual):
        """정확도를 계산합니다."""
        if pd.isna(predicted) and pd.isna(actual):
            return True
        
        if pd.isna(predicted) or pd.isna(actual):
            return False
        
        # 텍스트 정규화
        norm_pred = self._normalize_text(predicted)
        norm_actual = self._normalize_text(actual)
        
        # 정확한 매칭
        if norm_pred == norm_actual:
            return True
        
        # 유사한 의미 매칭
        similar_mappings = {
            # 파손 관련
            ('없음', '있음'): False,  # 반대 의미
            ('있음', '없음'): False,  # 반대 의미
            
            # 재료 관련 - 혼합 소재 허용
            ('플라스틱', '플라스틱,금속'): True,  # 부분 포함
            ('플라스틱', '플라스틱,섬유'): True,  # 부분 포함
            ('금속', '플라스틱,금속'): True,      # 부분 포함
            
            # 오염도 관련 - 유사한 수준 허용
            ('깨끗', '보통'): True,   # 유사한 수준
            ('보통', '더러움'): False, # 반대 의미
            ('깨끗', '더러움'): False, # 반대 의미
        }
        
        # 유사성 체크
        for (val1, val2), is_similar in similar_mappings.items():
            if (norm_pred == val1 and norm_actual == val2) or (norm_pred == val2 and norm_actual == val1):
                return is_similar
        
        return False

    def _extract_ai_confidence(self, ai_details, agent_name):
        """AI 상세 정보에서 confidence를 안전하게 추출합니다."""
        try:
            if agent_name in ai_details:
                agent_data = ai_details[agent_name]
                if isinstance(agent_data, dict):
                    return agent_data.get('confidence', '불명')
                else:
                    return '불명'
            return '불명'
        except:
            return '불명'
    
    def _extract_ai_notes(self, ai_details, agent_name):
        """AI 상세 정보에서 notes를 안전하게 추출합니다."""
        try:
            if agent_name in ai_details:
                agent_data = ai_details[agent_name]
                if isinstance(agent_data, dict):
                    return str(agent_data.get('notes', ''))[:100]
                else:
                    return str(agent_data)[:100]
            return ''
        except:
            return ''

def main():
    """메인 실행 함수"""
    evaluator = ImprovedCSVDonationEvaluator()
    
    # CSV 파일 경로
    csv_path = "Data_to_evaluate.csv"
    
    if not Path(csv_path).exists():
        print(f"❌ CSV 파일을 찾을 수 없습니다: {csv_path}")
        return
    
    # 평가 실행
    results, accuracy = evaluator.evaluate_from_csv(csv_path)
    
    if results:
        print(f"\n🎉 평가 완료! 전체 정확도: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
