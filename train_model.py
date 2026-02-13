import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib # 모델 저장을 위한 라이브러리

# ---------------------------------------------------------
# 1. 데이터 로드 및 전처리 (과거 데이터 통합 포함)
# ---------------------------------------------------------

def process_and_merge_data():
    print("데이터 통합 작업 시작")

    # (1) 메인 데이터 (2020-2023) 로드
    df_recent = pd.read_csv('data/Biogas_Model_Input_2020_2023.csv', encoding='utf-8')
    cols = ['Food_Waste_Amount_Ton', 'Biogas_Production_TOE']
    for col in cols:
        if df_recent[col].dtype == object:
            df_recent[col] = df_recent[col].str.replace(',', '').astype(float)

    # (2) 과거 데이터 (1996-2019) 로드 및 전처리
    try:
        df_hist = pd.read_csv('data/Waste_History_1996_2019.csv', encoding='cp949')

        # 지역명 매핑 (긴 이름 -> 짧은 이름)
        region_map = {
            '서울특별시': '서울', '부산광역시': '부산', '대구광역시': '대구', '인천광역시': '인천',
            '광주광역시': '광주', '대전광역시': '대전', '울산광역시': '울산', '세종특별자치시': '세종',
            '경기도': '경기', '강원도': '강원', '충청북도': '충북', '충청남도': '충남',
            '전라북도': '전북', '전라남도': '전남', '경상북도': '경북', '경상남도': '경남',
            '제주특별자치도': '제주'
        }
        
        # 필요한 행만 필터링 (음식물류 폐기물 + 분리배출 합산)
        # 과거에는 혼합배출(음식물류 폐기물)이었고, 최근엔 분리배출 -> 두 항목을 합쳐야 전체 발생량이 됨
        target_items = ['음식물류폐기물', '음식물류 폐기물 분리배출']
        df_hist = df_hist[df_hist['항목별'].isin(target_items) & df_hist['구분'].isin(region_map.keys())]

        # 연도 컬럼만 선택 ('1996 년' -> '1996')
        year_cols = [c for c in df_hist.columns if '년' in c]

        # 결측치 0 처리
        df_hist[year_cols] = df_hist[year_cols].fillna(0)

        # 지역별, 연도별 합계 계산 (두 항목 합산)
        df_grouped = df_hist.groupby('구분')[year_cols].sum().reset_index()

        # Long Format으로 변환 (연도, 지역, 발생량)
        df_long = df_grouped.melt(id_vars='구분', var_name='Year_Raw', value_name='Amount_TonPerDay')
        df_long['Year'] = df_long['Year_Raw'].str.replace(' 년', '').astype(int)
        df_long['Region'] = df_long['구분'].map(region_map)

        # 단위 변환: 톤/일 -> 톤/년 (윤년 고려)
        def get_days(y): return 366 if (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0) else 365
        df_long['Food_Waste_Amount_Ton'] = df_long['Amount_TonPerDay'] * df_long['Year'].apply(get_days)

        # 필요한 컬럼만 선택
        df_hist_clean = df_long[['Year', 'Region', 'Food_Waste_Amount_Ton']].sort_values(['Region', 'Year'])

        print("과거 데이터(1996-2019) 처리 완료")

        # (3) 두 데이터 합치기 (시각화용 전체 히스토리)
        # 최근 데이터에서 필요한 컬럼만 가져오기
        df_recent_subset = df_recent[['Year', 'Region', 'Food_Waste_Amount_Ton']]

        # 위아래로 붙이기
        df_full_history = pd.concat([df_hist_clean, df_recent_subset], ignore_index=True).sort_values(['Region', 'Year'])
        
        return df_recent, df_full_history
    
    except FileNotFoundError:
        print("과거 데이터 파일이 없습니다.")
        return df_recent, df_recent[['Year', 'Region', 'Food_Waste_Amount_Ton']]
    
# ---------------------------------------------------------
# 2. 모델 학습
# ---------------------------------------------------------

def train_model(df):
    # Feature & Target
    features = [
        'Food_Waste_Amount_Ton',
        'Capacity_Manure_TonPerDay',
        'Capacity_FoodWaste_TonPerDay',
        'Capacity_Combined_TonPerDay',
        'Capacity_SewageSludge_TonPerDay'
    ]
    target = 'Biogas_Production_TOE'

    # 시설 용량 데이터가 없는 과거 데이터는 학습에서 제외 (Biogas Target도 없음)
    # 따라서 학습은 'df_recent'만 사용
    X = df[features]
    y = df[target]

    # 학습/테스트 데이터 분리(성능 평가를 위해)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)

    # 모델 성능 평가
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'모델 성능 - MAE: {mae:.2f}, R2: {r2:.2f}')

    return model, mae, r2

# =========================================================
# 실행
# =========================================================

if __name__ == "__main__":
    # 1. 데이터 통합
    df_recent, df_full_history = process_and_merge_data()

    # 2. 모델 학습 (최근 데이터 사용)
    rf_model, mae, r2 = train_model(df_recent)
    print("예측 모델 학습 완료")

    # 3. 저장
    joblib.dump(rf_model, "biogas_rf_model.pkl")

    # 모델 성능 지표 저장
    metrics = {'mae': mae, 'r2': r2}
    joblib.dump(metrics, 'model_metrics.pkl')
    print(f'성능 지표 저장 완료: {metrics}')
    
    # 웹사이트에서 쓸 데이터 저장
    # (1) 예측용 데이터 (시설용량 포함된 최근 데이터)
    df_recent.to_pickle("processed_data_recent.pkl")
    # (2) 시각화용 데이터 (1996 - 2023 전체 히스토리)
    df_full_history.to_pickle("processed_data_history.pkl")

    print("모든 파일 저장 완료!")
    print("'streamlit run app.py'를 실행하세요!")
