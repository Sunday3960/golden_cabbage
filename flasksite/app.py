# from flask import Flask, render_template, request, jsonify
# from prophet import Prophet
# import pandas as pd

# app = Flask(__name__)

# # 예측 모델 로드
# df = pd.read_csv('/home/user/golden_cabbage/flasksite/static/full_month.csv')
# #df.columns = ['ds', 'y']  # Prophet은 'ds'와 'y' 컬럼 이름을 사용합니다.
# df['ds'] = pd.to_datetime(df['ds'])  # 날짜 형식으로 변환
# holidays = pd.DataFrame({
#     'holiday': '김장철',
#     'ds': pd.to_datetime(['2013-11', '2014-11', '2015-11','2016-11', '2017-11', '2018-11', '2019-11','2020-11', '2021-11', '2022-11', '2023-11']),
#     'lower_window': 0,
#     'upper_window': 30,
# })
# # 모델 훈련
# m = Prophet(holidays=holidays,yearly_seasonality=True)
# m.fit(df)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     # 입력받은 월 정보
#     input_month = request.form.get('month')

#     try:
#         # 입력받은 월을 날짜 형식으로 변환 (예: '2023-10')
#         input_date = pd.to_datetime(input_month + '-01')

#         # 미래 예측을 위한 데이터프레임 생성
#         future = m.make_future_dataframe(periods=24, freq='M')  # 12개월 예측
#         forecast = m.predict(future)
        
#         # 예측 결과에서 입력된 월의 다음 달 예측값 추출
#         prediction_date = input_date + pd.DateOffset(months=1)  # 다음 달 첫째 날
#         predicted_price = forecast[forecast['ds'] == prediction_date]['yhat'].values

#         if len(predicted_price) > 0:
#             return jsonify({'predicted_price': predicted_price[0]})
#         else:
#             return jsonify({'error': 'No prediction available for this date.'}), 404

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, render_template, request, jsonify
from prophet import Prophet
import pandas as pd

app = Flask(__name__)

# 데이터 로드 및 모델 학습
df = pd.read_csv('/home/user/golden_cabbage/flasksite/static/full_month.csv')
df['ds'] = pd.to_datetime(df['ds'])
holidays = pd.DataFrame({
    'holiday': '김장철',
    'ds': pd.to_datetime(['2013-11', '2014-11', '2015-11', '2016-11', '2017-11', '2018-11', '2019-11', '2020-11', '2021-11', '2022-11', '2023-11']),
    'lower_window': 0,
    'upper_window': 30,
})
m = Prophet(holidays=holidays, yearly_seasonality=True)
m.fit(df)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_month = request.form.get('month')
    try:
        input_date = pd.to_datetime(input_month + '-01')
        future = m.make_future_dataframe(periods=24, freq='M')
        forecast = m.predict(future)
        prediction_date = input_date + pd.DateOffset(months=1)
        predicted_price = forecast[forecast['ds'] == prediction_date]['yhat'].values
        if len(predicted_price) > 0:
            return jsonify({'predicted_price': predicted_price[0]})
        else:
            return jsonify({'error': 'No prediction available for this date.'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/price-trend-data')
def price_trend_data():
    return jsonify({'dates': df['ds'].dt.strftime('%Y-%m').tolist(), 'prices': df['y'].tolist()})

@app.route('/forecast-data')
def forecast_data():
    future = m.make_future_dataframe(periods=24, freq='M')
    forecast = m.predict(future)
    return jsonify({
        'dates': forecast['ds'].dt.strftime('%Y-%m').tolist(),
        'predicted_prices': forecast['yhat'].tolist()
    })

@app.route('/feature-importance-data')
def feature_importance_data():
    return jsonify({
        'features': ['feature1', 'feature2', 'feature3'],
        'importances': [0.6, 0.3, 0.1]
    })

if __name__ == '__main__':
    app.run(debug=True)
