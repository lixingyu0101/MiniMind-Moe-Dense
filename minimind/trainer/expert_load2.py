import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["font.size"] = 12


def load_experiment(dir_path):
    """
    读取某一个实验目录下的所有 json
    """
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
# 修改成你的真实路径
# =========================
exp_configs = {
    "No Load (α=0.0)": "../out_exp1_1",
    "Weak Load (α=0.01)": "../out_exp2_1",
    "Strong Load (α=0.05)": "../out_exp3",
}

results = {}
for name, path in exp_configs.items():
    results[name] = load_experiment(path)


# =========================
# Figure 1: CV 曲线
# =========================
plt.figure()
for name, (steps, cvs, _, _) in results.items():
    plt.plot(steps, cvs, label=name)
plt.xlabel("Training Step")
plt.ylabel("Coefficient of Variation (CV)")
plt.title("Expert Load Balance (CV ↓ better)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("cv_curve.png")
plt.show()


# =========================
# Figure 2: Max / Mean 曲线
# =========================
plt.figure()
for name, (steps, _, ratios, _) in results.items():
    plt.plot(steps, ratios, label=name)
plt.xlabel("Training Step")
plt.ylabel("Max / Mean Load")
plt.title("Expert Load Skewness")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("max_mean_curve.png")
plt.show()


# =========================
# Figure 3: Aux Loss 曲线
# =========================
plt.figure()
for name, (steps, _, _, aux_losses) in results.items():
    plt.plot(steps, aux_losses, label=name)
plt.xlabel("Training Step")
plt.ylabel("Auxiliary Load Loss")
plt.title("Aux Loss vs Load Balancing Strength")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("aux_loss_curve.png")
plt.show()
