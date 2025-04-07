from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

input_features = [
    'pelvic_incidence', 'pelvic_tilt', 'lumbar_lordosis_angle',
    'sacral_slope', 'pelvic_radius', 'degree_spondylolisthesis'
]

# Load models and preprocessing tools
lr_model = joblib.load('models/lr_spinal.pkl')
svc_model = joblib.load('models/spine_svc.pkl')
rf_model = joblib.load('models/spine_rf.pkl')
scaler = joblib.load('models/scaler_std.pkl')
power_transformer = joblib.load('models/power_transformer.pkl')

# Labels for prediction display
label_map = {
    0: ("Hernia", "orange", "Signs of herniated disc detected. Consider a clinical consultation for further evaluation."),
    1: ("Normal", "green", "Your spine alignment appears normal. Keep maintaining a healthy lifestyle!"),
    2: ("Spondylolisthesis", "red", "Warning: Potential spondylolisthesis detected. Please consult a spine specialist promptly.")
}

# Slip category based on degree
def categorize_slip(degree):
    if degree < 6:
        return 'Normal'
    elif degree < 17:
        return 'Mild'
    else:
        return 'Severe'

# Full preprocessing logic matching training
def preprocess_input(input_data: dict, model_type: str = 'lr_svc') -> np.ndarray:
    df = pd.DataFrame([input_data])[input_features].astype(float)

    # Clip negative values (as per training logic)
    df['pelvic_tilt'] = df['pelvic_tilt'].clip(lower=0)
    df['sacral_slope'] = df['sacral_slope'].clip(lower=0)
    df['degree_spondylolisthesis'] = df['degree_spondylolisthesis'].clip(lower=0)

    # Power transform pelvic_tilt and pelvic_radius
    df[['pelvic_tilt', 'pelvic_radius']] = power_transformer.transform(df[['pelvic_tilt', 'pelvic_radius']])

    # Slip category one-hot encoding
    slip_cat = categorize_slip(df['degree_spondylolisthesis'].values[0])
    df['slip_Mild'] = int(slip_cat == 'Mild')
    df['slip_Normal'] = int(slip_cat == 'Normal')
    df['slip_Severe'] = int(slip_cat == 'Severe')

    if model_type == 'lr_svc':
        # Create log version of degree_spondylolisthesis
        df['degree_spondylolisthesis_log'] = np.log1p(df['degree_spondylolisthesis'])
        df.drop(columns=['degree_spondylolisthesis'], inplace=True)

        final_features = [
            'pelvic_incidence', 'pelvic_tilt', 'lumbar_lordosis_angle',
            'sacral_slope', 'pelvic_radius', 'slip_Mild', 'slip_Normal',
            'slip_Severe', 'degree_spondylolisthesis_log'
        ] 

        return scaler.transform(df[final_features])

    else:  # For Random Forest (no scaling, no log transform)
        final_features = [
            'pelvic_incidence', 'pelvic_tilt', 'lumbar_lordosis_angle',
            'sacral_slope', 'pelvic_radius', 'degree_spondylolisthesis',
            'slip_Mild', 'slip_Normal', 'slip_Severe'
        ]
        return df[final_features].values

@app.route('/')
def index():
    return render_template('index.html', input_features={})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect user input from form
        input_data = {feature: float(request.form[feature]) for feature in input_features}
        model_choice = request.form['model']

        if model_choice == 'lr':
            X_processed = preprocess_input(input_data, 'lr_svc')
            prediction = lr_model.predict(X_processed)[0]
            confidence = lr_model.predict_proba(X_processed).max() * 100

        elif model_choice == 'svc':
            X_processed = preprocess_input(input_data, 'lr_svc')
            prediction = svc_model.predict(X_processed)[0]
            scores = svc_model.decision_function(X_processed)
            confidence = scores[0][prediction] if scores.ndim > 1 else scores[0]
            confidence = min(abs(confidence), 10.0) / 10.0 * 100

        elif model_choice == 'rf':
            X_processed = preprocess_input(input_data, 'rf')
            prediction = rf_model.predict(X_processed)[0]
            confidence = rf_model.predict_proba(X_processed).max() * 100

        else:
            raise ValueError("Invalid model selected.")

        result, color, message = label_map[prediction]

        return render_template(
            'index.html',
            prediction_text=result,
            prediction_color=color,
            prediction_message=message,
            confidence_score=f"{confidence:.2f}",
            input_features=input_data
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f"Error: {str(e)}",
            prediction_color="black",
            prediction_message="Something went wrong. Please double-check your input values.",
            input_features=request.form
        )

if __name__ == '__main__':
    app.run(debug=True)
