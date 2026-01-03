import re
import matplotlib.pyplot as plt

# -----------------------
# 日志解析函数
# -----------------------
def load_train_loss(log_file):
    """提取sft阶段 Loss"""
    steps, losses = [], []
    step_counter = 0
    with open(log_file, 'r') as f:
        for line in f:
            if 'loss:' in line and 'Epoch' in line and '[VAL]' not in line:
                m = re.search(r'loss:([0-9.]+)', line)
                if m:
                    step_counter += 1
                    steps.append(step_counter)
                    losses.append(float(m.group(1)))
    return steps, losses

def load_val(log_file):
    """提取验证阶段 Loss 和 PPL"""
    epochs, val_losses, val_ppls = [], [], []
    with open(log_file, 'r') as f:
        for line in f:
            if '[VAL]' in line:
                m_epoch = re.search(r'Epoch \[(\d+)\]', line)
                m_loss = re.search(r'loss: ([0-9.]+)', line)
                m_ppl  = re.search(r'ppl: ([0-9.]+)', line)
                if m_epoch and m_loss and m_ppl:
                    epochs.append(int(m_epoch.group(1)))
                    val_losses.append(float(m_loss.group(1)))
                    val_ppls.append(float(m_ppl.group(1)))
    return epochs, val_losses, val_ppls

# -----------------------
# 读取日志
# -----------------------
dense_train_steps, dense_train_loss = load_train_loss('sft_dense.log')
moe_train_steps, moe_train_loss     = load_train_loss('sft_moe.log')

dense_epochs, dense_val_loss, dense_val_ppl = load_val('sft_dense.log')
moe_epochs, moe_val_loss, moe_val_ppl       = load_val('sft_moe.log')

# -----------------------
# 绘图设置
# -----------------------
plt.rcParams.update({
    "font.size": 10,
    "figure.figsize": (12, 8),
    "axes.grid": True
})

fig, axes = plt.subplots(3, 1, sharex=False)

# 子图1：Train Loss vs Step
axes[0].plot(dense_train_steps, dense_train_loss, label='Dense', color='tab:blue')
axes[0].plot(moe_train_steps, moe_train_loss, label='MoE', color='tab:orange')
axes[0].set_xlabel('Training Steps')
axes[0].set_ylabel('Train Loss')
axes[0].set_title('Training Loss vs Step')
axes[0].legend()

# 子图2：Validation Loss vs Epoch
axes[1].plot(dense_epochs, dense_val_loss, label='Dense', marker='o', color='tab:blue')
axes[1].plot(moe_epochs, moe_val_loss, label='MoE', marker='o', color='tab:orange')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Validation Loss')
axes[1].set_title('Validation Loss vs Epoch')
axes[1].legend()

# 子图3：Validation PPL vs Epoch
axes[2].plot(dense_epochs, dense_val_ppl, label='Dense', marker='s', color='tab:blue')
axes[2].plot(moe_epochs, moe_val_ppl, label='MoE', marker='s', color='tab:orange')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Validation PPL')
axes[2].set_title('Validation PPL vs Epoch')
axes[2].legend()

plt.tight_layout()
plt.savefig('SFT_loss_comparison.png', dpi=300)
plt.show()
