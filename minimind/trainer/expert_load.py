import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (15, 4.5)
plt.rcParams["font.size"] = 12


def load_experiment(dir_path):
    files = sorted(glob.glob(os.path.join(dir_path, "*.json")))
    steps, cvs, ratios, aux_losses = [], [], [], []

    for f in files:
        with open(f, "r") as fp:
            data = json.load(fp)
        steps.append(data["step"])
        cvs.append(data["cv"])
        ratios.append(data["max_mean_ratio"])
        aux_losses.append(data["aux_loss"])

    return (
        np.array(steps),
        np.array(cvs),
        np.array(ratios),
        np.array(aux_losses),
    )


# =========================
# 修改为你的真实实验路径
# =========================
exp_configs = {
    "No Load (α=0.0)": "../out_exp1_1",
    "Weak Load (α=0.01)": "../out_exp2_1",
    "Strong Load (α=0.05)": "../out_exp3",
}

results = {
    name: load_experiment(path)
    for name, path in exp_configs.items()
}

# =========================
# Combined Figure
# =========================
fig, axes = plt.subplots(1, 3)

# -------- Subplot 1: CV --------
for name, (steps, cvs, _, _) in results.items():
    axes[0].plot(steps, cvs, label=name)

axes[0].set_title("Expert Load Balance (CV ↓)")
axes[0].set_xlabel("Training Step")
axes[0].set_ylabel("CV")
axes[0].grid(True)

# -------- Subplot 2: Max / Mean --------
for name, (steps, _, ratios, _) in results.items():
    axes[1].plot(steps, ratios, label=name)

axes[1].set_title("Expert Load Skewness")
axes[1].set_xlabel("Training Step")
axes[1].set_ylabel("Max / Mean")
axes[1].grid(True)

# -------- Subplot 3: Aux Loss --------
for name, (steps, _, _, aux_losses) in results.items():
    axes[2].plot(steps, aux_losses, label=name)

axes[2].set_title("Auxiliary Load Loss")
axes[2].set_xlabel("Training Step")
axes[2].set_ylabel("Aux Loss")
axes[2].grid(True)

# -------- Legend & Layout --------
fig.legend(
    results.keys(),
    loc="upper center",
    ncol=3,
    frameon=False,
)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig("moe_load_analysis_combined.png", dpi=300)
plt.show()
