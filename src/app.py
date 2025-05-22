from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Cargar modelo y transformador
model = pickle.load(open("src/xgboost_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_input = np.array(features).reshape(1, -1)
    transformed_input = transformer.transform(final_input)
    prediction = model.predict(transformed_input)
    return render_template("index.html", prediction_text=f"Predicción: {'Diabético' if prediction[0]==1 else 'No diabético'}")

if __name__ == "__main__":
    app.run(debug=True)