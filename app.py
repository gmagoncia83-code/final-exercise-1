"""SLA Breach Prediction - Student Service Request Monitoring (Section B).

Streamlit deployment of the midterm Decision Tree (target: SLA_Breached).
"""
import datetime as dt

import pandas as pd
import streamlit as st

from modules import dashboard, data_processing as dp
from modules import exploratory_analysis as eda
from modules import prediction as pr

st.set_page_config(page_title="SAD Predictive System", layout="wide")

PAGES = ["Home", "Dataset", "Data Analysis", "Prediction", "Dashboard", "System Design"]
menu = st.sidebar.selectbox("Module", PAGES)
role = st.sidebar.selectbox(
    "Signed in as (demo role)",
    ["Student Services Staff", "Administrator", "Student / Requester"],
)
st.sidebar.caption("Predictions are decision support only. Use synthetic or "
                   "authorized data for public deployment.")


# ---------- cached loaders ----------
@st.cache_resource
def load_model():
    return pr.load_artifacts()


@st.cache_data
def load_default_data():
    raw = pd.read_csv(pr.DATA_PATH)
    return dp.prepare(raw)


def current_data():
    """Uploaded dataset (if accepted) else the bundled synthetic dataset."""
    if "df" in st.session_state:
        return st.session_state["df"], st.session_state["log"], st.session_state["source"]
    df, log = load_default_data()
    return df, log, "Bundled synthetic sample (data/student_service_requests.csv)"


pipe, meta = load_model()
df, clean_log, source = current_data()
if "decisions" not in st.session_state:
    st.session_state["decisions"] = []


# ---------- pages ----------
if menu == "Home":
    st.title("Student Service Request Monitoring - SLA Breach Prediction")
    st.write(
        "This app deploys the Decision Tree from the Midterm Laboratory Examination. "
        "It predicts whether a request is likely to **breach its service-level "
        "agreement (SLA)** using information known **at submission time only**."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Model version", meta["model_version"])
    c2.metric("Test recall (breach class)", f"{meta['metrics']['recall']:.2f}")
    c3.metric("Requests loaded", len(df))
    st.subheader("Modules")
    st.markdown(
        "1. **Dataset** - upload a CSV, schema checks, null/duplicate summary\n"
        "2. **Data Analysis** - statistics, class distribution, interactive charts\n"
        "3. **Prediction** - enter a new request and get a risk estimate\n"
        "4. **Dashboard** - KPIs, risk/review queue, recommended follow-up\n"
        "5. **System Design** - requirements, actors, use case diagram, workflow"
    )
    st.info("A high-risk flag means *review*, never automatic denial of a service.")
    st.caption(f"Data source: {source}")

elif menu == "Dataset":
    st.title("Dataset Upload and Validation")
    st.caption("Expected columns: " + ", ".join(dp.REQUIRED_COLUMNS))
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read the file as CSV: {e}")
            st.stop()
        if raw.empty:
            st.error("The uploaded file has no rows.")
            st.stop()
        ok, missing, extra = dp.validate_schema(raw)
        if not ok:
            st.error("Schema check failed. Missing columns: " + ", ".join(missing))
            st.stop()
        if extra:
            st.warning("Extra columns ignored: " + ", ".join(extra))
        st.success("Schema check passed.")
        st.subheader("Preview")
        st.dataframe(raw.head(20))
        q = dp.quality_summary(raw)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", q["rows"])
        c2.metric("Columns", q["columns"])
        c3.metric("Duplicate rows", q["duplicate_rows"])
        c4.metric("Duplicate Request_IDs", q["duplicate_ids"])
        st.subheader("Null summary")
        st.dataframe(q["nulls"])
        new_df, log = dp.prepare(raw)
        if new_df.empty or dp.labeled_subset(new_df).empty:
            st.error("After cleaning there are no completed requests with a known "
                     "SLA outcome, so the dataset cannot be analysed.")
        else:
            st.subheader("Cleaning log")
            for line in log:
                st.write("- " + line)
            if st.button("Use this dataset in all modules"):
                st.session_state.update(df=new_df, log=log, source=f"Uploaded: {uploaded.name}")
                st.rerun()
    else:
        st.info("No file uploaded - modules use the bundled synthetic sample.")
        st.dataframe(df.head(20))
        if "df" in st.session_state and st.button("Switch back to bundled sample"):
            for k in ("df", "log", "source"):
                st.session_state.pop(k, None)
            st.rerun()
    with st.expander("Cleaning log for the data currently in use"):
        for line in clean_log:
            st.write("- " + line)

elif menu == "Data Analysis":
    st.title("Exploratory Data Analysis")
    st.caption(f"Data source: {source}")
    lab = dp.labeled_subset(df)
    tab1, tab2, tab3 = st.tabs(["Statistics", "Class distribution", "Charts"])
    with tab1:
        st.dataframe(eda.descriptive_stats(df))
    with tab2:
        st.dataframe(eda.class_distribution(df))
        st.plotly_chart(eda.fig_class_distribution(df))
        st.caption(f"Modeling subset: {len(lab)} completed requests with known SLA outcome "
                   "(open and cancelled requests excluded).")
    with tab3:
        st.plotly_chart(eda.fig_requests_by_service(df))
        st.plotly_chart(eda.fig_breach_by_office(df))
        st.plotly_chart(eda.fig_breach_by_queue(df))
        st.plotly_chart(eda.fig_breach_by_weekday_channel(df))
        st.plotly_chart(eda.fig_monthly_volume(df))

elif menu == "Prediction":
    st.title("SLA Breach Prediction")
    st.caption("Uses submission-time information only. No outcome fields are used.")
    cats = meta["categories"]
    with st.form("predict_form"):
        c1, c2 = st.columns(2)
        service = c1.selectbox("Service type", cats["Service_Type"])
        requester = c1.selectbox("Requester type", cats["Requester_Type"])
        channel = c1.selectbox("Requested channel", cats["Requested_Channel"])
        priority = c1.selectbox("Priority", cats["Priority"])
        office = c2.selectbox("Assigned office", cats["Assigned_Office"])
        docs = c2.selectbox("Required documents complete?", cats["Required_Documents_Complete"])
        qmax = int(max(meta["numeric_ranges"]["Queue_Length"][1], 40))
        queue_len = c2.number_input("Queue length at submission", 0, qmax, 8)
        smax = int(max(meta["numeric_ranges"]["SLA_Days"][1], 30))
        sla_days = c2.number_input("SLA target (days)", 1, smax, 5)
        sub_date = c1.date_input("Submission date", dt.date.today())
        go = st.form_submit_button("Predict")
    if go:
        row = pr.build_input_row(service, requester, channel, priority, office,
                                 docs, queue_len, sla_days, sub_date)
        res = pr.predict_one(pipe, row)
        st.subheader("Result")
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted class", "Likely BREACH" if res["label"] == 1 else "Likely within SLA")
        if res["probability"] is not None:
            c2.metric("Breach probability", f"{res['probability']:.0%}")
            c3.metric("Risk level", res["risk"])
        st.write("**Input sent to the model**")
        st.dataframe(row)
        st.caption(f"Model version {meta['model_version']} | Trained {meta['trained_at']} | "
                   f"Prediction timestamp {res['timestamp']}")
        st.warning("Human review required: this prediction does not authorize rejecting "
                   "or denying any request. " + dashboard.FOLLOW_UP.get(res["risk"], ""))

    st.subheader("Model metrics (held-out 20% test set)")
    m = meta["metrics"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{m['accuracy']:.2f}")
    c2.metric("Precision", f"{m['precision']:.2f}")
    c3.metric("Recall", f"{m['recall']:.2f}")
    c4.metric("F1", f"{m['f1']:.2f}")
    c5.metric("Majority baseline", f"{m['baseline_accuracy']:.2f}")
    cm = pd.DataFrame(m["confusion_matrix"], index=["Actual: Met SLA", "Actual: Breach"],
                      columns=["Pred: Met SLA", "Pred: Breach"])
    st.write("**Confusion matrix**")
    st.dataframe(cm)
    st.write("**Feature importance**")
    fi = pd.Series(meta["feature_importance"], name="Importance").round(3)
    st.bar_chart(fi)
    with st.expander("Limitations"):
        for item in pr.LIMITATIONS:
            st.write("- " + item)
    with st.expander("Check consistency with the midterm notebook"):
        st.write("Recomputes metrics on the same 80/20 split (random_state=42) and "
                 "compares them with the saved ones.")
        if st.button("Run consistency check"):
            st.dataframe(pr.verify_consistency(pipe, meta, load_default_data()[0]))

elif menu == "Dashboard":
    st.title("Decision-Support Dashboard")
    st.caption(f"Data source: {source}")
    if role == "Student / Requester":
        st.error("This dashboard is for Student Services Staff and Administrators.")
        st.stop()
    k = dashboard.kpis(df)
    cols = st.columns(len(k))
    for col, (name, val) in zip(cols, k.items()):
        col.metric(name, val)
    scored = dashboard.score_requests(df, pipe)

    left, right = st.columns(2)
    left.subheader("Office workload")
    left.dataframe(dashboard.office_workload(df))
    right.subheader("Breach rate by queue category")
    right.plotly_chart(eda.fig_breach_by_queue(df))

    st.subheader("Risk / review queue (open requests)")
    thr = st.slider("Flag open requests with breach probability at or above", 0.1, 0.95, 0.5, 0.05)
    queue = dashboard.review_queue(scored, thr)
    st.write(f"{len(queue)} open request(s) flagged for human review.")
    st.dataframe(queue)

    st.subheader("Interpretation")
    st.write(
        f"Top drivers in the model: **{', '.join(list(meta['feature_importance'])[:3])}**. "
        "Flagged requests are candidates for review - check document completeness and "
        "office workload before acting. Missing or unusual data lowers reliability."
    )

    st.subheader("Record final decision")
    if role == "Student Services Staff" and len(queue):
        with st.form("decision_form"):
            rid = st.selectbox("Request", queue["Request_ID"])
            action = st.selectbox("Human decision", [
                "Prioritize / expedite", "Request missing documents",
                "Reassign to another staff member", "No action needed"])
            note = st.text_input("Reviewer note")
            if st.form_submit_button("Save decision"):
                st.session_state["decisions"].append({
                    "Request_ID": rid, "Decision": action, "Note": note,
                    "Recorded_by": role,
                    "Recorded_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
                st.success("Decision recorded.")
    elif role == "Administrator":
        st.info("Administrators can view results and decisions but do not record case decisions.")
    if st.session_state["decisions"]:
        log_df = pd.DataFrame(st.session_state["decisions"])
        st.dataframe(log_df)
        st.download_button("Download decision log (CSV)", log_df.to_csv(index=False),
                           "decision_log.csv", "text/csv")

elif menu == "System Design":
    st.title("SAD Integration")
    lab = dp.labeled_subset(df)
    overall = lab[dp.TARGET].mean()
    by_office = eda.breach_rate_by(df, "Assigned_Office").sort_values("Breach_Rate_%", ascending=False)
    by_queue = eda.breach_rate_by(df, "Queue_Category").set_index("Queue_Category")["Breach_Rate_%"]
    by_docs = eda.breach_rate_by(df, "Required_Documents_Complete").set_index(
        "Required_Documents_Complete")["Breach_Rate_%"]
    top = by_office.iloc[0]

    st.subheader("Functional requirements (evidence from the data)")
    st.markdown(
        f"1. The system shall display the overall SLA breach rate and request volume - "
        f"*evidence: overall breach rate is {overall:.1%} across {len(lab)} completed requests.*\n"
        f"2. The system shall report breach rate per office - *evidence: "
        f"{top['Assigned_Office']} is highest at {top['Breach_Rate_%']}%.*\n"
        f"3. The system shall flag open requests with high queue length - *evidence: "
        f"High-queue breach rate is {by_queue.get('High', float('nan'))}% vs "
        f"{by_queue.get('Low', float('nan'))}% for Low.*\n"
        f"4. The system shall warn staff when required documents are incomplete - *evidence: "
        f"breach rate {by_docs.get('No', float('nan'))}% (incomplete) vs "
        f"{by_docs.get('Yes', float('nan'))}% (complete).*\n"
        f"5. The system shall predict breach risk at submission and route high-risk requests "
        f"to human review - *evidence: model recall {meta['metrics']['recall']:.2f}, "
        f"accuracy {meta['metrics']['accuracy']:.2f} vs baseline {meta['metrics']['baseline_accuracy']:.2f}.*"
    )

    st.subheader("Non-functional requirements")
    st.markdown(
        "1. **Performance** - the saved model is loaded once (cached); predictions return in under 2 seconds.\n"
        "2. **Privacy and security** - only synthetic or authorized data is deployed; no credentials "
        "or real student records are stored in the repository.\n"
        "3. **Transparency / usability** - every prediction shows model version, timestamp, "
        "limitations and a human-review notice."
    )

    st.subheader("Actors and permissions")
    st.table(pd.DataFrame([
        {"Actor": "Student / Requester", "View results": "No", "Record final decision": "No",
         "Role": "Submits requests"},
        {"Actor": "Student Services Staff", "View results": "Yes", "Record final decision": "Yes",
         "Role": "Reviews flagged requests, decides follow-up"},
        {"Actor": "Administrator", "View results": "Yes", "Record final decision": "No",
         "Role": "Monitors SLA performance and workload"},
    ]))

    st.subheader("Updated use case diagram")
    st.image("assets/use_case_diagram.png")

    st.subheader("Analytical process / decision workflow")
    st.graphviz_chart("""
    digraph G {
      rankdir=LR; node [shape=box, style=rounded];
      A [label="Request submitted"]; B [label="Validate & clean data"];
      C [label="Submission-time features"]; D [label="Decision Tree\\n(saved pipeline)"];
      E [label="Breach probability"]; F [label="Risk / review queue"];
      G [label="Staff review"]; H [label="Final human decision\\n(recorded)"];
      A -> B -> C -> D -> E -> F -> G -> H;
    }
    """)
