# SLA Breach Prediction - Student Service Request Monitoring (Section B)

Streamlit deployment of the Midterm Laboratory Examination (Systems Analysis and Design, BS Information Systems).
Target: `SLA_Breached` (1 if Processing_Days > SLA_Days, completed requests only).
Model: Decision Tree inside a preprocessing pipeline (saved with joblib, random_state=42, 80/20 split).

## Modules
Home | Dataset | Data Analysis | Prediction | Dashboard | System Design

## Run locally
```
pip install -r requirements.txt
python train_model.py        # one time: creates models/trained_model.joblib
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
Push to GitHub -> https://share.streamlit.io -> New app -> pick repo/branch -> entry point `app.py` -> Deploy.

## Reusing your actual midterm work
1. Replace `data/student_service_requests.csv` with your midterm dataset (same columns).
2. Match your notebook's model settings in `build_pipeline()` (`modules/prediction.py`).
3. Run `python train_model.py`, then commit the new `models/` files.
4. Use the "Check consistency with the midterm notebook" button on the Prediction page.

## Data and ethics
Only synthetic data is included. Predictions are decision support; authorized staff make the final decision.
No passwords or real student records are stored in this repository.

## Files
`app.py`, `requirements.txt`, `modules/{data_processing,exploratory_analysis,prediction,dashboard}.py`,
`models/trained_model.joblib`, `assets/use_case_diagram.png`, plus `train_model.py`,
`make_sample_data.py`, `make_use_case_diagram.py`.

## Revised midterm notebook
`notebooks/SectionB_Midterm_Revised.ipynb` runs Parts I-IV (cleaning, features, EDA, Decision Tree, evaluation) with
submission-time features only and the same settings as the app. Its results match the app's saved metrics.
Write your own answers/interpretations in the markdown cells (Q10-Q14).

## Docs
`docs/ACCEPTANCE_CHECKLIST.md`, `docs/SUBMISSION_CHECKLIST.md`, `docs/REQUIREMENTS.md`, `docs/DEPLOYMENT_REPORT.md`.
