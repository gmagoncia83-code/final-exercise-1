"""Draws assets/use_case_diagram.png (updated UML use case diagram)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

fig, ax = plt.subplots(figsize=(11, 7))
ax.set_xlim(0, 11); ax.set_ylim(0, 7); ax.axis("off")
ax.add_patch(Rectangle((2.6, 0.3), 5.8, 6.4, fill=False, lw=2))
ax.text(5.5, 6.45, "Student Service Request Monitoring System", ha="center", weight="bold")

cases = {  # name: (y, is_new)
    "Submit Service Request": (5.6, False),
    "Process Request": (4.7, False),
    "Upload & Validate Dataset": (3.8, True),
    "View SLA Performance": (2.9, True),
    "Review Delay Risk": (2.0, True),
    "Record Final Decision": (1.1, True),
}
pos = {}
for name, (y, new) in cases.items():
    ax.add_patch(Ellipse((5.5, y), 3.6, 0.75, fc="#dff3e3" if new else "#ffffff", ec="black"))
    ax.text(5.5, y, name, ha="center", va="center", fontsize=9)
    pos[name] = (5.5, y)

def actor(x, y, label):
    ax.plot([x], [y + 0.45], "o", ms=14, mfc="white", mec="black")
    ax.plot([x, x], [y + 0.3, y - 0.2], "k-")
    ax.plot([x - 0.3, x + 0.3], [y + 0.15, y + 0.15], "k-")
    ax.plot([x, x - 0.25], [y - 0.2, y - 0.7], "k-")
    ax.plot([x, x + 0.25], [y - 0.2, y - 0.7], "k-")
    ax.text(x, y - 1.0, label, ha="center", fontsize=9)

actor(1.2, 5.3, "Student /\nRequester")
actor(1.2, 3.0, "Student Services\nStaff")
actor(9.8, 3.6, "Administrator")

def link(a, b, ax_pos, side):
    x, y = pos[b]
    x += -1.8 if side == "L" else 1.8
    ax.plot([ax_pos[0], x], [ax_pos[1], y], "k-", lw=1)

link("", "Submit Service Request", (1.5, 5.3), "L")
for c in ["Process Request", "Upload & Validate Dataset", "Review Delay Risk", "Record Final Decision"]:
    link("", c, (1.5, 3.0), "L")
for c in ["View SLA Performance", "Review Delay Risk", "Upload & Validate Dataset"]:
    link("", c, (9.5, 3.6), "R")

ax.text(5.5, 0.05, "Green = new analytics use cases added after the midterm findings",
        ha="center", fontsize=8, style="italic")
fig.savefig("assets/use_case_diagram.png", dpi=150, bbox_inches="tight")
