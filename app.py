from flask import Flask, render_template, request
import pandas as pd
import pickle
import warnings
import sys
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore")

app = Flask(__name__)

# === Paths to your pickles ===
MODEL_PATH = "./churn_model.pkl"
SCALER_PATH = "./standard_scalar.pkl"

# Load model & scaler
try:
    model = pickle.load(open(MODEL_PATH, "rb"))
except Exception as e:
    print("ERROR loading model:", e, file=sys.stderr)
    model = None

try:
    scaler = pickle.load(open(SCALER_PATH, "rb"))
except Exception as e:
    print("WARNING loading scaler:", e, file=sys.stderr)
    scaler = None

# Model features
model_features = getattr(model, "feature_names_in_", None)
if model_features is not None:
    model_features = list(model_features)
else:
    model_features = []

# Preprocessing config
nominal_cols = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'PaperlessBilling', 'PaymentMethod'
]

contract_order = [['Month-to-month', 'One year', 'Two year']]
sim_map = {'Jio': 0, 'Airtel': 1, 'Vi': 2, 'BSNL': 3}
ordinal_encoder = OrdinalEncoder(categories=contract_order, handle_unknown='use_encoded_value', unknown_value=-1)


def preprocess_input(form_dict):
    df = pd.DataFrame([form_dict])
    df = df.rename(columns={c: c.strip() for c in df.columns})
    df = df.applymap(lambda v: v.strip() if isinstance(v, str) else v)

    # Convert numeric
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c])
        except:
            pass

    # Map sim
    if 'sim' in df.columns:
        df['sim'] = df['sim'].map(sim_map).fillna(0)

    # Contract ordinal
    if 'Contract' in df.columns:
        try:
            df['Contract'] = ordinal_encoder.transform(df[['Contract']])
        except Exception:
            df['Contract'] = -1

    # One-hot
    cols_to_dummy = [col for col in nominal_cols if col in df.columns]
    if cols_to_dummy:
        df = pd.get_dummies(df, columns=cols_to_dummy, drop_first=False)

    # Ensure charges
    df['MonthlyCharges'] = df.get('MonthlyCharges', df.get('MonthlyCharges_qt_outl', 0))
    df['TotalCharges'] = df.get('TotalCharges', df.get('TotalCharges_qt_outl', 0))

    # Scaling
    scale_cols = ['MonthlyCharges', 'TotalCharges']
    for sc in scale_cols:
        df[sc] = pd.to_numeric(df.get(sc, 0), errors='coerce').fillna(0)
    if scaler is not None and hasattr(scaler, "mean_"):
        try:
            df[scale_cols] = scaler.transform(df[scale_cols])
        except:
            df[scale_cols] = (df[scale_cols] - df[scale_cols].mean()) / (df[scale_cols].std().replace(0, 1))
    else:
        df[scale_cols] = (df[scale_cols] - df[scale_cols].mean()) / (df[scale_cols].std().replace(0, 1))

    # Align columns
    if model_features:
        for col in model_features:
            if col not in df.columns:
                df[col] = 0
        df_final = df.reindex(columns=model_features, fill_value=0)
    else:
        df_final = df.copy()

    return df_final


@app.route('/')
def home():
    return render_template('index.html', title="Customer Churn Prediction")


@app.route('/about')
def about():
    return render_template('about.html', title="About the Model")


@app.route('/developer')
def developer():
    return render_template('developer.html', title="Developer")


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('result.html', prediction_text="❌ Model not loaded. Check server logs.")

    try:
        form_data = {k.strip(): v.strip() for k, v in request.form.items()}
        X_final = preprocess_input(form_data)

        pred = model.predict(X_final)[0]

        probability = None
        try:
            if hasattr(model, "predict_proba"):
                probability = float(model.predict_proba(X_final)[0][1])
        except:
            probability = None

        if int(pred) == 1:
            result = f"❌ Customer likely to CHURN (prob={probability:.2f})" if probability else "❌ Customer likely to CHURN"
        else:
            result = f"✅ Customer likely to STAY (prob={probability:.2f})" if probability else "✅ Customer likely to STAY"

        return render_template('result.html', prediction_text=result)

    except Exception as e:
        print("Unhandled error in /predict:", e, file=sys.stderr)
        return render_template('result.html', prediction_text="⚠️ An unexpected error occurred.")


if __name__ == "__main__":
    app.run(debug=True)
