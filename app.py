
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the saved machine learning pipeline
model = joblib.load("house_price_model.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            # Read the values entered on the website
            house = {
                "area": float(request.form["area"]),
                "bedrooms": int(request.form["bedrooms"]),
                "bathrooms": int(request.form["bathrooms"]),
                "stories": int(request.form["stories"]),
                "mainroad": request.form["mainroad"],
                "guestroom": request.form["guestroom"],
                "basement": request.form["basement"],
                "hotwaterheating": request.form["hotwaterheating"],
                "airconditioning": request.form["airconditioning"],
                "parking": int(request.form["parking"]),
                "prefarea": request.form["prefarea"],
                "furnishingstatus": request.form["furnishingstatus"],
            }

            if house["area"] <= 0:
                raise ValueError("Area must be greater than zero.")

            if any(house[k] < 0 for k in
                   ["bedrooms", "bathrooms", "stories", "parking"]):
                raise ValueError("Counts cannot be negative.")

            # Preserve the feature order and names used during training
            input_data = pd.DataFrame([house])
            prediction = float(model.predict(input_data)[0])

        except (ValueError, KeyError) as exc:
            error = str(exc) or "Please check the values and try again."

    return render_template(
        "index.html",
        prediction=prediction,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
