"""Draws assets/decision_workflow.png (decision-support workflow)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

steps = ["Request\nsubmitted", "Validate &\nclean data", "Submission-time\nfeatures only",
         "Saved pipeline\n(Decision Tree)", "Breach probability\n& risk level",
         "Risk / review\nqueue", "Staff review\n(human)", "Final decision\nrecorded"]
fig, ax = plt.subplots(figsize=(14, 3.2)); ax.axis("off"); ax.set_xlim(0, 14); ax.set_ylim(0, 3)
for i, s in enumerate(steps):
    x = 0.2 + i * 1.7
    ax.add_patch(FancyBboxPatch((x, 1.0), 1.4, 1.0, boxstyle="round,pad=0.05",
                                fc="#dff3e3" if i >= 5 else "#eef3fb", ec="black"))
    ax.text(x + 0.7, 1.5, s, ha="center", va="center", fontsize=8)
    if i < len(steps) - 1:
        ax.annotate("", xy=(x + 1.65, 1.5), xytext=(x + 1.45, 1.5), arrowprops=dict(arrowstyle="->"))
ax.text(7, 0.4, "A high-risk flag triggers human review - never automatic denial of a service.",
        ha="center", style="italic", fontsize=9)
fig.savefig("assets/decision_workflow.png", dpi=150, bbox_inches="tight")
