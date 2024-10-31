from flask import Flask, render_template, request, jsonify
from prophet import Prophet
import pandas as pd

app = Flask(__name__)

# 데이터 로드 및 모델 학습
df = pd.read_csv('/home/user/golden_cabbage/flasksite/static/full_month_price.csv')
df['ds'] = pd.to_datetime(df['ds'])
df['y'] = df['y'].astype(float)  # 가격 데이터 형식 확인

holidays = pd.DataFrame({
    'holiday': '김장철',
    'ds': pd.to_datetime(['2013-11-25', '2014-11-25', '2015-11-25', '2016-11-25', '2017-11-25', 
                          '2018-11-25', '2019-11-25', '2020-11-25', '2021-11-25', 
                          '2022-11-25', '2023-11-25']),
    'lower_window': 0,
    'upper_window': 30,
})

m = Prophet(daily_seasonality=True)
m.fit(df)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    input_date_str = request.form.get('date')  # 'month'에서 'date'로 변경
    try:
        input_date = pd.to_datetime(input_date_str)
        
        # 예측을 위한 미래 날짜 생성 (입력된 날짜로부터 100일 예측)
        future = m.make_future_dataframe(periods=100, freq='D')  # 예측 범위를 100일로 증가
        forecast = m.predict(future)

        # 예측 날짜를 계산합니다
        prediction_date = input_date  # 입력된 날짜로 예측
        
        # 예측 결과에서 입력된 날짜 이후의 값 찾기
        predicted_price = forecast[forecast['ds'] >= prediction_date]  # 입력된 날짜 이후의 예측만 필터링
        
        print("예측 날짜:", prediction_date)
        print("예측 결과:", forecast[['ds', 'yhat']].tail(10))  # 최근 예측 결과 확인
        
        if not predicted_price.empty:
            return jsonify({'predicted_price': predicted_price.iloc[0]['yhat'], 'forecast': predicted_price[['ds', 'yhat']].to_dict(orient='records')})
        else:
            return jsonify({'error': 'No prediction available for this date.'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/price-trend-data')
def price_trend_data():
    return jsonify({'dates': df['ds'].dt.strftime('%Y-%m-%d').tolist(), 'prices': df['y'].tolist()})

@app.route('/forecast-data')
def forecast_data():
    future = m.make_future_dataframe(periods=30, freq='D')  # 100일 예측
    forecast = m.predict(future)
    return jsonify({
        'dates': forecast['ds'].dt.strftime('%Y-%m-%d').tolist(),
        'predicted_prices': forecast['yhat'].tolist()
    })

@app.route('/feature-importance-data')
def feature_importance_data():
    return jsonify({
        'features': ['feature1', 'feature2', 'feature3'],
        'importances': [0.6, 0.3, 0.1]
    })

if __name__ == '__main__':
    app.run(debug=True, port=5555)
