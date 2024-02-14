import joblib
import numpy as np

from flask import Flask
from flask import jsonify

app = Flask(__name__)

# construir endpoint, postman para prueba o usar navegador
@app.route('/predict', methods=['GET'])
def predict():
    X_test = np.array([6.594444821,1.479555538,1.616463184,1.53352356,0.796666503,0.635422587,0.362012237,0.315963835,2.277026653])
    prediction = model.predict(X_test.reshape(1,-1))
    return jsonify({'prediccion' : list(prediction)})

if __name__ == "__main__":
    model = joblib.load('./models/best_model.pkl')
    app.run(port=8000)
