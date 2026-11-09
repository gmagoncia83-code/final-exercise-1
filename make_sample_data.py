"""Generate a SYNTHETIC student_service_requests.csv (no real student records).

Replace data/student_service_requests.csv with your instructor-provided file
from the midterm if you have it.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 420

services = ["Records", "Certification", "IT Support", "Scholarship Inquiry", "Clearance"]
base = {"Records": 3, "Certification": 4, "IT Support": 1, "Scholarship Inquiry": 5, "Clearance": 3}
sla = {"Records": 5, "Certification": 5, "IT Support": 2, "Scholarship Inquiry": 7, "Clearance": 4}
office_for = {"Records": "Registrar", "Certification": "Registrar", "IT Support": "IT Office",
              "Scholarship Inquiry": "Scholarship Office", "Clearance": "Student Affairs"}
offices = sorted(set(office_for.values()))

svc = rng.choice(services, n, p=[0.28, 0.2, 0.22, 0.14, 0.16])
office = np.array([office_for[s] if rng.random() < 0.85 else rng.choice(offices) for s in svc])
req_type = rng.choice(["Student", "Faculty", "Staff"], n, p=[0.75, 0.1, 0.15])
channel = rng.choice(["Portal", "Email", "Walk-in"], n, p=[0.55, 0.2, 0.25])
priority = rng.choice(["Low", "Normal", "High"], n, p=[0.25, 0.6, 0.15])
queue = np.clip(rng.poisson(9, n) + (office == "Registrar") * 4, 0, 30)
docs = rng.choice(["Yes", "No"], n, p=[0.78, 0.22])
dates = pd.to_datetime("2026-01-05") + pd.to_timedelta(rng.integers(0, 270, n), unit="D")

proc = (np.array([base[s] for s in svc]) + 0.18 * queue + (docs == "No") * 2.5
        + np.where(priority == "High", -1.0, np.where(priority == "Low", 1.0, 0))
        + (channel == "Walk-in") * -0.5 + (office == "Registrar") * 0.8
        + rng.normal(0, 1.4, n))
proc = np.clip(np.round(proc), 1, None)

status = rng.choice(["Completed", "Pending", "Cancelled"], n, p=[0.86, 0.09, 0.05])
proc = np.where(status == "Completed", proc, np.nan)
dates = pd.Series(dates)
pend = status == "Pending"
dates[pend] = pd.to_datetime("2026-10-20") + pd.to_timedelta(rng.integers(0, 20, pend.sum()), unit="D")

df = pd.DataFrame({
    "Request_ID": [f"REQ{1000 + i}" for i in range(n)],
    "Service_Type": svc, "Requester_Type": req_type,
    "Submission_Date": dates.dt.strftime("%Y-%m-%d"),
    "Requested_Channel": channel, "Priority": priority, "Assigned_Office": office,
    "Queue_Length": queue, "Required_Documents_Complete": docs,
    "Processing_Days": proc, "SLA_Days": [sla[s] for s in svc], "Status": status,
    "Reopened": np.where((status == "Completed") & (rng.random(n) < 0.08), "Yes", "No"),
    "Satisfaction_Score": np.where((status == "Completed") & (rng.random(n) < 0.7),
                                   rng.integers(1, 6, n), np.nan),
})

# A few deliberate data-quality problems so the cleaning log has something to show
df.loc[5, "Requested_Channel"] = "walk-in"
df.loc[17, "Priority"] = " high"
df.loc[40, "Service_Type"] = "it support"
df.loc[60, "Queue_Length"] = -3
df.loc[90, "Submission_Date"] = "2026-13-45"
df = pd.concat([df, df.iloc[[10, 25, 80]]], ignore_index=True)

df.to_csv("data/student_service_requests.csv", index=False)
print("Saved", len(df), "rows")
