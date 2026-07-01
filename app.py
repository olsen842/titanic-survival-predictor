from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        age = float(request.form.get("age"))
        fare = float(request.form.get("fare"))
        pclass = int(request.form.get("pclass"))
        alone = int(request.form.get("alone"))


        who = request.form.get("who")
        who_man = 1 if who == "man" else 0
        who_woman = 1 if who == "woman" else 0



        embarked = request.form.get("embarked")
        embarked_Q = 1 if embarked == "Q" else 0
        embarked_S = 1 if embarked == "S" else 0


        input_data = np.array([[age, fare, pclass, alone, who_man, who_woman, embarked_Q, embarked_S]])
        prediction = model.predict(input_data)
        result = "🚢 Survived" if prediction[0] == 1 else "❌ Did not survive"
        return render_template("index.html", result=result)

    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=True)
