import mlflow.sklearn
import pandas as pd
from flask import Flask, jsonify, request

import mlflow

# Set the tracking server to local server
mlflow.set_tracking_uri('http://localhost:5001')

app = Flask(__name__)

logged_model = 'runs:/238469551abb4712be7f73e7bb21d00e/iris_rf_model'


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']

    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    # Predict on a Pandas DataFrame.
    predictions = loaded_model.predict(pd.DataFrame(data))

    return jsonify(predictions.tolist())


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
