from flask import Flask, render_template, request, jsonify
from prophet import Prophet
import pandas as pd

app = Flask(__name__)

# 예측 모델 로드
# 적절한 데이터로 모델을 훈련하세요.
# 예를 들어, 다음과 같은 형식의 데이터프레임을 사용한다고 가정합니다.
df = pd.read_csv('/home/user/golden_cabbage/flasksite/static/full_month.csv')
# df.columns = ['ds', 'y']  # Prophet은 'ds'와 'y' 컬럼 이름을 사용합니다.

# 모델 훈련
m = Prophet(yearly_seasonality=True)
m.fit(df)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 입력받은 날짜 정보
    input_month = request.form.get('month')
    # 예측을 위한 데이터 처리 (입력 데이터에 기반하여 예측할 날짜를 설정)
    # 다음 예시는 1개월 후의 예측을 위한 것으로 수정할 수 있습니다.
    
    # 예측하기
    future = m.make_future_dataframe(periods=30)  # 예를 들어, 30일 후까지 예측
    forecast = m.predict(future)
    
    # 예측 결과에서 필요한 데이터 추출
    predicted_price = forecast['yhat'].iloc[-1]  # 마지막 예측값

    return jsonify({'predicted_price': predicted_price})

if __name__ == '__main__':
    app.run(debug=True)
