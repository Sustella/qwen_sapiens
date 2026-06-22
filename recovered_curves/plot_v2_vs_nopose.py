"""Reconstruct training-loss + val curves for qwen_autism_v2 (pose) vs
qwen_autism_no_pose, parsed from their accelerate run logs.

Per-step training loss / LR comes from each "epoch N  step M  loss ..." line;
val metrics come from "gen-val step N" lines (both scripts log 20-sample
mid-training val). no_pose's val is byte-identical across all measurements
(LR collapsed to 0 at step 8 — see lr plot); a tiny jitter is added so the
6 measurement points are visually distinguishable instead of stacking on
top of each other.
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent

# ── Parsers ──────────────────────────────────────────────────────────────────
TRAIN_RE_V2 = re.compile(
    r"epoch (\d+)\s+step (\d+)\s+loss ([\d.eE+-]+)\s+"
    r"backbone_lr ([\d.eE+-]+)\s+proj_lr ([\d.eE+-]+)"
)
TRAIN_RE_NP = re.compile(
    r"epoch (\d+)\s+step (\d+)\s+loss ([\d.eE+-]+)\s+lr ([\d.eE+-]+)"
)
VAL_RE = re.compile(
    r"gen-val step (\d+):\s+strict=([\d.]+)\s+f1=([\d.]+)\s+"
    r"hamming=([\d.]+)\s+bal_acc=([\d.]+)"
)


def parse_v2(log_paths):
    train = []  # (step, loss, backbone_lr, proj_lr)
    val = []    # (step, strict, f1, ham, bal)
    seen_steps = set()
    seen_val = set()
    for p in log_paths:
        for line in Path(p).read_text().splitlines():
            m = TRAIN_RE_V2.search(line)
            if m:
                step = int(m.group(2))
                if step in seen_steps:
                    continue
                seen_steps.add(step)
                train.append((step, float(m.group(3)),
                              float(m.group(4)), float(m.group(5))))
                continue
            m = VAL_RE.search(line)
            if m:
                step = int(m.group(1))
                if step in seen_val:
                    continue
                seen_val.add(step)
                val.append((step, float(m.group(2)), float(m.group(3)),
                            float(m.group(4)), float(m.group(5))))
    train.sort(); val.sort()
    return train, val


def parse_nopose(log_paths):
    train = []
    val = []
    seen_steps = set()
    seen_val = set()
    for p in log_paths:
        for line in Path(p).read_text().splitlines():
            m = TRAIN_RE_NP.search(line)
            if m:
                step = int(m.group(2))
                if step in seen_steps:
                    continue
                seen_steps.add(step)
                train.append((step, float(m.group(3)), float(m.group(4))))
                continue
            m = VAL_RE.search(line)
            if m:
                step = int(m.group(1))
                if step in seen_val:
                    continue
                seen_val.add(step)
                val.append((step, float(m.group(2)), float(m.group(3)),
                            float(m.group(4)), float(m.group(5))))
    train.sort(); val.sort()
    return train, val


V2_LOGS = [
    "/orcd/compute/ppliang/001/qwen_autism_v2/logs/run_20260522_180928.log",
    "/orcd/compute/ppliang/001/qwen_autism_v2/logs/run_20260522_221045.log",
    "/orcd/compute/ppliang/001/qwen_autism_v2/logs/run_20260523_003335.log",
]
NP_LOGS = [
    "/orcd/compute/ppliang/001/qwen_autism_no_pose/logs/run_20260521_085501.log",
]

v2_train, v2_val = parse_v2(V2_LOGS)
np_train, np_val = parse_nopose(NP_LOGS)

print(f"v2: {len(v2_train)} train steps, {len(v2_val)} val measurements")
print(f"no_pose: {len(np_train)} train steps, {len(np_val)} val measurements")

# ── Unpack ───────────────────────────────────────────────────────────────────
v2_steps      = np.array([t[0] for t in v2_train])
v2_loss       = np.array([t[1] for t in v2_train])
v2_backbonelr = np.array([t[2] for t in v2_train])
v2_projlr     = np.array([t[3] for t in v2_train])
v2_val_steps  = np.array([v[0] for v in v2_val])
v2_val_strict = np.array([v[1] for v in v2_val])
v2_val_f1     = np.array([v[2] for v in v2_val])
v2_val_ham    = np.array([v[3] for v in v2_val])
v2_val_bal    = np.array([v[4] for v in v2_val])

np_steps   = np.array([t[0] for t in np_train])
np_loss    = np.array([t[1] for t in np_train])
np_lr      = np.array([t[2] for t in np_train])

# Match no_pose val to pose val cadence: keep only steps that pose also
# measured at (10, 20, 30). pose only saves epoch-boundary checkpoints, so
# we can't recover its 5/15/25 vals; align both curves on the same x-ticks.
_pose_val_set = set(int(v[0]) for v in v2_val)
np_val_aligned = [v for v in np_val if int(v[0]) in _pose_val_set]
np_v_steps  = np.array([v[0] for v in np_val_aligned])
np_v_strict = np.array([v[1] for v in np_val_aligned])
np_v_f1     = np.array([v[2] for v in np_val_aligned])
np_v_ham    = np.array([v[3] for v in np_val_aligned])
np_v_bal    = np.array([v[4] for v in np_val_aligned])

EPOCH_BOUNDARIES = [10.5, 20.5]  # step counts at epoch ends


def add_epoch_lines(ax):
    for x in EPOCH_BOUNDARIES:
        ax.axvline(x, color="lightgray", linestyle=":", linewidth=0.8, zorder=0)


# ── Plot 1: training loss ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(np_steps, np_loss, "o-", color="tab:orange", label="no_pose", linewidth=1.5, markersize=4)
ax.plot(v2_steps, v2_loss, "o-", color="tab:blue",   label="pose", linewidth=1.5, markersize=4)
add_epoch_lines(ax)
ax.text(5.5,  ax.get_ylim()[1]*0.97, "epoch 1", ha="center", va="top", fontsize=8, color="gray")
ax.text(15.5, ax.get_ylim()[1]*0.97, "epoch 2", ha="center", va="top", fontsize=8, color="gray")
ax.text(25.5, ax.get_ylim()[1]*0.97, "epoch 3", ha="center", va="top", fontsize=8, color="gray")
ax.set_xlabel("global optimizer step  (1 step = grad_accum=8 iters)")
ax.set_ylabel("training loss")
ax.set_title("Training loss — qwen_autism_v2 (pose) vs qwen_autism_no_pose")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "v2_vs_nopose_train_loss.png", dpi=130)
plt.close(fig)

# ── Plot 2: learning rate (log y) ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
# Replace exact-zero LRs with a floor so they're visible on the log axis,
# but mark them clearly.
def lr_for_log(x, floor=1e-9):
    return np.where(x <= 0, floor, x)

ax.semilogy(np_steps, lr_for_log(np_lr),         "o-", color="tab:orange", label="no_pose lr", linewidth=1.5, markersize=4)
ax.semilogy(v2_steps, lr_for_log(v2_backbonelr), "o-", color="tab:blue",   label="pose backbone_lr", linewidth=1.5, markersize=4)
ax.semilogy(v2_steps, lr_for_log(v2_projlr),     "s--", color="tab:cyan",  label="pose proj_lr",     linewidth=1.5, markersize=4)
# Annotate the lr-collapse point
collapse_step = int(np_steps[np.where(np_lr == 0)[0][0]])
ax.annotate(f"no_pose lr → 0 at step {collapse_step}",
            xy=(collapse_step, 1e-9), xytext=(collapse_step + 1, 1e-7),
            fontsize=9, color="tab:orange",
            arrowprops=dict(arrowstyle="->", color="tab:orange", lw=1))
add_epoch_lines(ax)
ax.set_xlabel("global optimizer step")
ax.set_ylabel("learning rate (log scale; 1e-9 floor = effective 0)")
ax.set_title("Learning rate schedule — no_pose cosine collapses to 0 after ~7 steps")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "v2_vs_nopose_lr.png", dpi=130)
plt.close(fig)

# ── Plot 3: val strict ───────────────────────────────────────────────────────
def val_plot(metric_name, np_y, v2_y, ylim=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(np_v_steps, np_y, "o-", color="tab:orange",
            label="no_pose",
            linewidth=1.5, markersize=6)
    ax.plot(v2_val_steps, v2_y, "o-", color="tab:blue",
            label="pose", linewidth=1.5, markersize=6)
    add_epoch_lines(ax)
    ax.set_xlabel("global optimizer step")
    ax.set_ylabel(f"val {metric_name}  (n=20)")
    ax.set_title(f"Validation {metric_name} — qwen_autism_v2 vs qwen_autism_no_pose")
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"v2_vs_nopose_val_{metric_name}.png", dpi=130)
    plt.close(fig)


val_plot("strict", np_v_strict, v2_val_strict, ylim=(0.20, 0.55))
val_plot("f1",     np_v_f1,     v2_val_f1,     ylim=(0.35, 0.85))
val_plot("hamming", np_v_ham,   v2_val_ham,    ylim=(0.70, 0.95))
val_plot("balanced_acc", np_v_bal, v2_val_bal,  ylim=(0.55, 0.90))

# ── Stacked figure: train loss + 4 val metrics, shared x-axis ────────────────
fig, axes = plt.subplots(5, 1, figsize=(9, 14), sharex=True)

ax = axes[0]
ax.plot(np_steps, np_loss, "o-", color="tab:orange", label="no_pose", linewidth=1.5, markersize=4)
ax.plot(v2_steps, v2_loss, "o-", color="tab:blue",   label="pose",    linewidth=1.5, markersize=4)
ax.set_ylabel("training loss")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)
for x in EPOCH_BOUNDARIES:
    ax.axvline(x, color="lightgray", linestyle=":", linewidth=0.8, zorder=0)
ylim_top = ax.get_ylim()[1]
ax.text(5.5,  ylim_top * 0.97, "epoch 1", ha="center", va="top", fontsize=8, color="gray")
ax.text(15.5, ylim_top * 0.97, "epoch 2", ha="center", va="top", fontsize=8, color="gray")
ax.text(25.5, ylim_top * 0.97, "epoch 3", ha="center", va="top", fontsize=8, color="gray")

panel_defs = [
    ("val balanced_acc (n=20)", np_v_bal,    v2_val_bal,    (0.55, 0.90)),
    ("val f1 (n=20)",            np_v_f1,     v2_val_f1,     (0.35, 0.85)),
    ("val hamming (n=20)",       np_v_ham,    v2_val_ham,    (0.70, 0.95)),
    ("val strict (n=20)",        np_v_strict, v2_val_strict, (0.20, 0.55)),
]
for ax, (ylabel, np_y, v2_y, ylim) in zip(axes[1:], panel_defs):
    ax.plot(np_v_steps,   np_y, "o-", color="tab:orange", label="no_pose", linewidth=1.5, markersize=6)
    ax.plot(v2_val_steps, v2_y, "o-", color="tab:blue",   label="pose",    linewidth=1.5, markersize=6)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    for x in EPOCH_BOUNDARIES:
        ax.axvline(x, color="lightgray", linestyle=":", linewidth=0.8, zorder=0)
    ax.legend(loc="best")

axes[-1].set_xlabel("global optimizer step  (1 step = grad_accum=8 iters)")
fig.suptitle("Training and validation curves — pose vs no_pose", y=0.995)
fig.tight_layout()
fig.savefig(OUT_DIR / "v2_vs_nopose_stacked.png", dpi=130)
plt.close(fig)

# ── Grid figure: train loss (full width) + 2x2 val metrics ───────────────────
fig = plt.figure(figsize=(12, 12))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])

ax_loss = fig.add_subplot(gs[0, :])
ax_loss.plot(np_steps, np_loss, "o-", color="tab:orange", label="no_pose", linewidth=1.5, markersize=4)
ax_loss.plot(v2_steps, v2_loss, "o-", color="tab:blue",   label="pose",    linewidth=1.5, markersize=4)
ax_loss.set_ylabel("training loss")
ax_loss.set_xlabel("global optimizer step  (1 step = grad_accum=8 iters)")
ax_loss.legend(loc="upper right")
ax_loss.grid(True, alpha=0.3)
for x in EPOCH_BOUNDARIES:
    ax_loss.axvline(x, color="lightgray", linestyle=":", linewidth=0.8, zorder=0)
ylim_top = ax_loss.get_ylim()[1]
ax_loss.text(5.5,  ylim_top * 0.97, "epoch 1", ha="center", va="top", fontsize=8, color="gray")
ax_loss.text(15.5, ylim_top * 0.97, "epoch 2", ha="center", va="top", fontsize=8, color="gray")
ax_loss.text(25.5, ylim_top * 0.97, "epoch 3", ha="center", va="top", fontsize=8, color="gray")

val_axes = [
    fig.add_subplot(gs[1, 0]),
    fig.add_subplot(gs[1, 1]),
    fig.add_subplot(gs[2, 0]),
    fig.add_subplot(gs[2, 1]),
]
for ax, (ylabel, np_y, v2_y, ylim) in zip(val_axes, panel_defs):
    ax.plot(np_v_steps,   np_y, "o-", color="tab:orange", label="no_pose", linewidth=1.5, markersize=6)
    ax.plot(v2_val_steps, v2_y, "o-", color="tab:blue",   label="pose",    linewidth=1.5, markersize=6)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    for x in EPOCH_BOUNDARIES:
        ax.axvline(x, color="lightgray", linestyle=":", linewidth=0.8, zorder=0)
    ax.legend(loc="best")
val_axes[2].set_xlabel("global optimizer step")
val_axes[3].set_xlabel("global optimizer step")

fig.suptitle("Training and validation curves — pose vs no_pose", y=0.995)
fig.tight_layout()
fig.savefig(OUT_DIR / "v2_vs_nopose_grid.png", dpi=130)
plt.close(fig)

print("\nSaved plots to:", OUT_DIR)
for p in sorted(OUT_DIR.glob("v2_vs_nopose_*.png")):
    print(" ", p.name)
