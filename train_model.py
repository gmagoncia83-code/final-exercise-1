"""Run ONCE locally to create models/trained_model.joblib and model_metadata.json.

    python train_model.py

Then commit both files. Streamlit Cloud only loads them; it never retrains.
"""
from modules.prediction import train_and_save

pipe, meta = train_and_save()
print("Saved models/trained_model.joblib")
for k, v in meta["metrics"].items():
    print(k, v)
print("Top features:", list(meta["feature_importance"].items())[:3])
