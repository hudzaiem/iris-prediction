import numpy as np
import pandas as pd
import pickle
import re
import os
from flask import Flask,request, url_for, redirect, render_template

model = pickle.load(open("model.pkl", "rb"))
app = Flask(__name__)
IMG_FOLDER = os.path.join('static','IMG')
app.config['UPLOAD_FOLDER'] = IMG_FOLDER

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods =["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    feature = [np.array(float_features)]
    prediction = model.predict(feature)
    image = prediction[0]+'.png'
    image = os.path.join(app.config['UPLOAD_FOLDER'], image)
    return render_template("index.html", prediction_text ="{}".format(prediction[0]), image = image)

if __name__ == "__main__":
    app.run(debug=True,host='127.0.0.1', port=8080)
