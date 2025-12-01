from flask import Flask, render_template, request
import pickle
import traceback

app = Flask(__name__)

# Load trained model
model_path = "models/Heat_Detection_Model.pkl"
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        print("Model loaded successfully!")
except Exception as e:
    model = None
    print("Failed to load model:")
    traceback.print_exc()


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            # Get numeric inputs
            activity = float(request.form["activity"])
            temperature = float(request.form["temperature"])

            # Get categorical inputs
            vulva_swelling = request.form["vulva_swelling"]
            heat_sign = request.form["heat_sign"]
            behavior_change = request.form["behavior_change"]

            # Map categorical values
            vulva_map = {"None": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
            heat_map = {"None": 0, "Mounting Other Sows": 1, "Allowing Mount": 2, "Standing Heat": 3}
            behavior_map = {"Normal": 0, "Aggression": 1, "Loss of Appetite": 2,
                            "Increased Vocalization": 3, "Restlessness": 4}

            # Convert to numeric
            vulva_num = vulva_map.get(vulva_swelling, 0)
            heat_num = heat_map.get(heat_sign, 0)
            behavior_num = behavior_map.get(behavior_change, 0)

            # Compute combined heat + vulva score
            heat_vulva_score = vulva_num + heat_num

            # Prepare input row for model
            input_row = [activity, temperature, behavior_num, heat_vulva_score]

            if model:
                raw_pred = model.predict([input_row])[0]
                # Convert 0/1 prediction to readable text
                prediction = "Heat Detected" if raw_pred == 1 else "No Heat Detected"
            else:
                error = "Model not loaded"

        except Exception as e:
            error = str(e)
            traceback.print_exc()

    return render_template("index.html", prediction=prediction, error=error)


if __name__ == "__main__":
    app.run(debug=True)
