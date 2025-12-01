import pickle
import pandas as pd
import os

# Load trained model
model_path = os.path.join("models", "Heat_Detection_Model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Function to make prediction
def predict_heat(input_data):
    """
    input_data: dict with keys:
        - activity_level (float)
        - temperature_c (float)
        - vulva_swelling (str: 'None', 'Mild', 'Moderate', 'Severe')
        - heat_sign (str: 'None', 'Mounting Other Sows', 'Allowing Mount', 'Standing Heat', 'Restlessness')
        - behavior_change (str: 'Normal', 'Aggression', 'Loss of Appetite', 'Increased Vocalization', 'Restlessness')
    """
    # Mapping categorical features
    vulva_map = {"None": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
    heat_map = {"None": 0, "Mounting Other Sows": 1, "Allowing Mount": 2, "Standing Heat": 3, "Restlessness": 0}
    behavior_map = {"Normal": 0, "Aggression": 1, "Loss of Appetite": 2, "Increased Vocalization": 3, "Restlessness": 4}

    # Process input
    vulva_num = vulva_map.get(input_data.get("vulva_swelling", "None"), 0)
    heat_num = heat_map.get(input_data.get("heat_sign", "None"), 0)
    behavior_num = behavior_map.get(input_data.get("behavior_change", "Normal"), 0)
    heat_vulva_score = vulva_num + heat_num

    # Prepare feature vector
    X = pd.DataFrame([{
        "activity_level": input_data.get("activity_level", 0),
        "temperature_c": input_data.get("temperature_c", 0),
        "behavior_num": behavior_num,
        "heat_vulva_score": heat_vulva_score
    }])

    # Predict
    prediction = model.predict(X)[0]  # 0 = Not in Heat, 1 = In Heat
    probability = model.predict_proba(X)[0][1]  # probability of being in heat

    return {"prediction": int(prediction), "probability": float(probability)}
