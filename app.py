import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------
# 데이터 로드
# ---------------------------------------------------------

@st.cache_resource
def load_resources():
    try:
        model = joblib.load("biogas_rf_model.pkl")
        # 두 개의 데이터 파일을 불러옵니다.
        df_recent = pd.read_pickle("processed_data_recent.pkl") # 예측용 (시설정보 포함)
        df_hist = pd.read_pickle("processed_data_history.pkl") # 시각화용 (1996 ~ 2023)
        return model, df_recent, df_hist
    except FileNotFoundError:
        return None, None, None
    
model, df_recent, df_hist = load_resources()

# ---------------------------------------------------------
# 메인 화면
# ---------------------------------------------------------

st.set_page_config(page_title="바이오에너지 예측", layout="wide")
st.title("🌱 음식물류 폐기물 기반 바이오에너지 예측 플랫폼")

if df_recent is None:
    st.error("데이터가 없습니다. 먼저 'train_model.py'를 실행해주세요.")
else:
    # 탭 구성
    tab1, tab2 = st.tabs(["음식물 쓰레기 발생량 분석 및 예측", "에너지 전환량 예측 시뮬레이터"])

    # =========================================================
    # TAB 1: 장기 추세 분석 (1996 ~ 2028)
    # =========================================================
    
    with tab1:
        st.subheader("지역별 음식물 쓰레기 발생량 추이 및 미래 예측")

        col1, col2 = st.columns([1, 3])

        with col1:
            # 지역 선택
            region_list = df_hist['Region'].unique().tolist()
            selected_region = st.selectbox("분석할 지역을 선택하세요: ", region_list)

            st.markdown("---")
            st.markdown('#### 미래 시나리오 (2024 ~ 2028)')
            scenario = st.radio(
                "예측 모드 선택: ",
                ['최근 추세 반영 (감소/정체)',
                 '장기 추세 반영 (과거 10년 회귀)']
            )

            st.caption("""
            * **최근 추세**: 코로나 시기(2020~23)의 감소세가 이어진다고 가정
            * **장기 추세**: 인구 증가 및 배달 문화 등 장기적인 상승 압력 반영
            """)
        
        with col2:
            # 선택 지역 데이터 필터링
            viz_df = df_hist[df_hist['Region'] == selected_region].sort_values('Year')

            # 미래 예측 로직
            last_year = 2023
            last_val = viz_df.iloc[-1]['Food_Waste_Amount_Ton']
            future_years = [2024, 2025, 2026, 2027, 2028]
            future_vals = []

            if "최근 추세" in scenario:
                # 최근 5년(2019~2023) 데이터로만 기울기 계산
                recent_data = viz_df[viz_df['Year'] >= 2019]
                reg = LinearRegression()
                reg.fit(recent_data[['Year']], recent_data['Food_Waste_Amount_Ton'])
                future_vals = reg.predict(np.array(future_years).reshape(-1, 1))

            else: # 장기 추세 (과거 10년: 2014 ~ 2023 반영)
                # 코로나 시기의 급격한 감소를 노이즈로 보고, 장기적인 힘을 반영
                long_data = viz_df[viz_df['Year'] >= 2014]
                reg = LinearRegression()
                reg.fit(long_data[['Year']], long_data['Food_Waste_Amount_Ton'])
                future_vals = reg.predict(np.array(future_years).reshape(-1, 1))
            
            # 그래프용 데이터 생성
            future_df = pd.DataFrame({
                'Year' : future_years,
                'Food_Waste_Amount_Ton' : future_vals,
                'Type' : ['Prediction'] * 5
            })

            # 2023년과 연결하기 위해 마지막 실제값 추가
            connect_row = pd.DataFrame({
                'Year' : [2023],
                'Food_Waste_Amount_Ton' : [last_val],
                'Type' : ['Prediction']
            })
            future_df = pd.concat([connect_row, future_df]).sort_values('Year')

            viz_df['Type'] = 'Actual'
            final_df = pd.concat([viz_df[['Year', 'Food_Waste_Amount_Ton', 'Type']], future_df])

            # 시각화
            fig = px.line(final_df, x= 'Year', y='Food_Waste_Amount_Ton', color='Type',
                          color_discrete_map={'Actual': '#1f77b4', 'Prediction' : '#ff7f0e'},
                          title=f"{selected_region} 음식물 쓰레기 발생량 (1997~2028)", markers=True)
            fig.update_traces(line=dict(width=3))
            fig.add_vrect(x0=2019.5, x1=2022.5, annotation_text="COVID-19",
                          annotation_position="top left", fillcolor="gray", opacity=0.1, line_width=0)
            st.plotly_chart(fig, use_container_width=True)

    # =========================================================
    # TAB 2: 에너지 예측 (기존 기능 유지)
    # =========================================================   
    with tab2:
        st.subheader("음식물 쓰레기 -> 바이오에너지 전환 예측")
        
        c1, c2 = st.columns(2)
        with c1:
            r_select = st.selectbox("지역 선택(시설 용량 자동 로드)", region_list, key='pred_region')
            # 2023년 실제 발생량 표시
            curr_waste = df_recent[(df_recent['Region']==r_select) & (df_recent['Year']==2023)]['Food_Waste_Amount_Ton'].values[0]
            st.metric("2023년 실제 발생량", f"{curr_waste:,.0f} 톤")

            input_val = st.number_input("투입할 쓰레기 양 (톤/년)", value=float(curr_waste))

        with c2:
            st.write('### 예측 결과')
            if st.button("계산하기"):
                # 해당 지역 시설 정보
                fac_info = df_recent[df_recent['Region'] == r_select].iloc[-1]

                input_data = pd.DataFrame([{
                    'Food_Waste_Amount_Ton': input_val,
                    'Capacity_Manure_TonPerDay': fac_info['Capacity_Manure_TonPerDay'],
                    'Capacity_FoodWaste_TonPerDay': fac_info['Capacity_FoodWaste_TonPerDay'],
                    'Capacity_Combined_TonPerDay': fac_info['Capacity_Combined_TonPerDay'],
                    'Capacity_SewageSludge_TonPerDay': fac_info['Capacity_SewageSludge_TonPerDay']
                }])

                pred = model.predict(input_data)[0]
                st.success(f"예상 바이오가스 생산량: {pred:,.2f} TOE")
                st.info(f"이는 약 {pred*4.5:,.0f}가구의 월간 전력 사용량에 해당합니다.")