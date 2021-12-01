  
from math import expm1

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from tensorflow import keras
app = Flask(__name__)

model = keras.models.load_model("my_h5_model.h5")
transformer = joblib.load("data_transformer.joblib")



@app.route("/", methods=["POST"])
def index():
    data = request.json
    df = pd.DataFrame(data, index=[0])
    prediction = model.predict(transformer.transform(df))
    predicted_price = expm1(prediction.flatten()[0])
    return jsonify({"price": str(predicted_price)})
    pass