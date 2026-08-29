from flask import Flask, request, render_template
import joblib
import pandas as pd
from datetime import datetime

# creation de l'application Flask

app = Flask(__name__)

#chargement du modèle de prédiction
model = joblib.load("models/house_price_model.pkl")
print(" Modèle chargé !")


@app.route('/' , methods=['GET', 'POST'])
def index():
    prediction = None
    error = None
    if request.method == 'POST':
        try:
            sqft_living = float(
                request.form["sqft_living"]
            )

            sqft_lot = float(
                request.form["sqft_lot"]
            )

            sqft_above = float(
                request.form["sqft_above"]
            )

            sqft_basement = float(
                request.form["sqft_basement"]
            )

            sqft_living15 = float(
                request.form["sqft_living15"]
            )

            sqft_lot15 = float(
                request.form["sqft_lot15"]
            )

            bedrooms = float(
                request.form["bedrooms"]
            )

            bathrooms = float(
                request.form["bathrooms"]
            )

            floors = float(
                request.form["floors"]
            )

            condition = float(
                request.form["condition"]
            )

            grade = float(
                request.form["grade"]
            )

            view = float(
                request.form["view"]
            )

            waterfront = int(
                request.form["waterfront"]
            )

            yr_built = float(
                request.form["yr_built"]
            )

            yr_renovated = float(
                request.form["yr_renovated"]
            )

            lat = float(
                request.form["lat"]
            )

            long = float(
                request.form["long"]
            )

            zipcode = request.form["zipcode"]


            # ==================================
            # DATE
            # ==================================

            date = datetime.strptime(
                request.form["date"],
                "%Y-%m-%d"
            )


            # ==================================
            # FEATURE ENGINEERING
            # ==================================

            age_maison = (
                2015 - yr_built
            )

            renovee = int(
                yr_renovated > 0
            )

            ratio_surface = (
                sqft_living /
                (sqft_lot + 1)
            )

            total_rooms = (
                bedrooms +
                bathrooms
            )

            sqft_per_room = (
                sqft_living /
                (total_rooms + 1)
            )

            grade_condition = (
                grade *
                condition
            )


            # ==================================
            # CRÉATION DU DATAFRAME
            # ==================================

            data = pd.DataFrame([{

                "sqft_living": sqft_living,

                "sqft_above": sqft_above,

                "sqft_basement": sqft_basement,

                "sqft_living15": sqft_living15,

                "sqft_lot15": sqft_lot15,

                "ratio_surface": ratio_surface,

                "sqft_per_room": sqft_per_room,

                "lat": lat,

                "long": long,

                "bedrooms": bedrooms,

                "bathrooms": bathrooms,

                "floors": floors,

                "condition": condition,

                "grade": grade,

                "view": view,

                "total_rooms": total_rooms,

                "age_maison": age_maison,

                "grade_condition": grade_condition,

                "waterfront": waterfront,

                "renovee": renovee,

                "year": date.year,

                "month": date.month,

                "zipcode": zipcode

            }])


            # ==================================
            # PRÉDICTION
            # ==================================

            prediction = model.predict(data)[0]
        except Exception as e:
            error = str(e)

    return render_template(
        'index.html',
        prediction=prediction,
        error=error
    )


if __name__ == '__main__':
    app.run(debug=True) 
    