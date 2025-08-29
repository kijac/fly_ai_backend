import streamlit as st
import traceback
from ai_agent.supervisor_agent import SupervisorAgent
from ai_agent.image_input import get_image_streamlit

def main():
    st.set_page_config(
        page_title="장난감 기부 판별 AI (고도화)",
        page_icon="🧸",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # 헤더
    st.title("🧸 장난감 기부 판별 AI (고도화)")
    st.markdown("---")
    st.markdown("### 📸 장난감 이미지를 업로드하면, AI가 기부 가능 여부와 처리 방법을 판별해드립니다.")
    
    # 병렬/순차 처리 선택 옵션
    processing_mode = st.sidebar.selectbox(
        "🔧 처리 모드 선택",
        ["병렬 처리 (빠름)", "순차 처리 (안정적)"],
        help="병렬 처리는 4개 AI 에이전트를 동시에 실행하여 속도를 높입니다."
    )
    
    # 이미지 업로드
    image = get_image_streamlit()
    
    if image:
        # 이미지 표시
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### 📷 업로드된 이미지")
            st.image(image, width=300)
        
        with col2:
            try:
                supervisor = SupervisorAgent()
                
                # 처리 모드에 따른 메시지
                if "병렬" in processing_mode:
                    with st.spinner("🤖 AI가 병렬로 이미지를 분석 중입니다... (4개 에이전트 동시 실행)"):
                        result = supervisor.process(image, image)  # 현재는 단일 이미지 사용
                else:
                    with st.spinner("🤖 AI가 순차적으로 이미지를 분석 중입니다..."):
                        result = supervisor.process_sequential(image, image)
                
                st.markdown("#### 🔍 AI 분석 결과")
                
                # 처리 시간 및 모드 표시
                processing_info = f"⏱️ 처리 시간: {result.get('processing_time', 0):.2f}초 | "
                processing_info += f"🚀 {'병렬 처리' if result.get('parallel_processing', False) else '순차 처리'}"
                st.info(processing_info)
                
                # 기부 가능 여부를 가장 먼저 표시
                if result["기부 가능 여부"] == "가능":
                    st.success("✅ **기부 가능한 장난감입니다!**")
                else:
                    st.error("❌ **기부가 어려운 장난감입니다**")
                    if result["기부 불가 사유"]:
                        st.warning(f"💡 사유: {result['기부 불가 사유']}")
                
                st.markdown("---")
                
                # 분석 상세 정보를 카드 형태로 표시
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("#### 🎯 장난감 정보")
                    st.info(f"**종류**: {result['장난감 종류']}")
                    st.info(f"**재료**: {result['재료']}")
                    st.info(f"**건전지**: {result['건전지 여부']}")
                    st.info(f"**크기**: {result['크기']}")
                
                with col_b:
                    st.markdown("#### 🔧 상태 및 처리")
                    
                    # 파손 상태에 따른 색상 구분
                    damage = result['파손']
                    if damage == "없음":
                        st.success(f"**파손 상태**: {damage} ✨")
                    elif "심각" in damage:
                        st.error(f"**파손 상태**: {damage} ⚠️")
                    else:
                        st.warning(f"**파손 상태**: {damage} 🔍")
                    
                    # 오염도 표시
                    soil = result['오염도']
                    if soil == "깨끗":
                        st.success(f"**오염도**: {soil} ✨")
                    elif soil == "더러움":
                        st.error(f"**오염도**: {soil} 🚫")
                    else:
                        st.warning(f"**오염도**: {soil} 🔍")
                    
                    # 수리/분해 정보
                    repair_info = result['수리/분해']
                    if "수리 불필요" in repair_info:
                        st.success(f"**처리 방법**: {repair_info} 🎉")
                    elif "수리 가능" in repair_info or "경미 수리" in repair_info:
                        st.warning(f"**처리 방법**: {repair_info} 🔧")
                    elif "분해" in repair_info or "업사이클" in repair_info:
                        st.info(f"**처리 방법**: {repair_info} ♻️")
                    else:
                        st.error(f"**처리 방법**: {repair_info} ❓")
                
                # 관찰사항 표시
                if result['관찰사항']:
                    st.markdown("---")
                    st.markdown("#### 📝 관찰사항")
                    st.info(result['관찰사항'])
                
                # 토큰 사용량 표시 (개발자용)
                if st.checkbox("🔧 개발자 정보 보기"):
                    st.markdown("---")
                    st.markdown("#### 🔧 토큰 사용량")
                    token_usage = result.get('토큰 사용량', {})
                    st.json(token_usage)
                        
            except Exception as e:
                st.error(f"❌ 오류가 발생했습니다: {str(e)}")
                st.code(traceback.format_exc())

if __name__ == "__main__":
    main()