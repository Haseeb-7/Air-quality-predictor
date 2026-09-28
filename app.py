from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained AdaBoost model
model = joblib.load("air_quality_model.plk")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    pm10 = float(request.form["pm10"])
    pm2_5 = float(request.form["pm2_5"])
    carbon_monoxide = float(request.form["carbon_monoxide"])
    nitrogen_dioxide = float(request.form["nitrogen_dioxide"])
    sulphur_dioxide = float(request.form["sulphur_dioxide"])
    ozone = float(request.form["ozone"])
    aerosol_optical_depth = float(request.form["aerosol_optical_depth"])
    dust = float(request.form["dust"])
    uv_index = float(request.form["uv_index"])

    features = [[
        pm10,
        pm2_5,
        carbon_monoxide,
        nitrogen_dioxide,
        sulphur_dioxide,
        ozone,
        aerosol_optical_depth,
        dust,
        uv_index
    ]]

    prediction = model.predict(features)

    if prediction[0] == "good":
        result = "Good"
    else:
        result = "Poor"

    return render_template(
        "index.html",
        prediction_text=f"The air quality is {result}"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)