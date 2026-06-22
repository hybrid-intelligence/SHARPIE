import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

# ── argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Plot benchmark percentiles from CSV")
parser.add_argument("csv_path", help="Path to merged CSV file")
parser.add_argument("--output", "-o", default="benchmark_percentiles.png",
                    help="Output image path (default: benchmark_percentiles.png)")
args = parser.parse_args()

CSV_PATH = args.csv_path
OUT_PATH = args.output
PERCENTILES       = [50,             95,     99]
PERCENTILE_LABELS = ["p50 (median)", "p95",  "p99"]

# One style (dash+marker) per percentile, same for every variable
LINESTYLES = {
    "p50 (median)": "-",
    "p95":          "--",
    "p99":          ":",
}
MARKERS = {
    "p50 (median)": "o",
    "p95":          "s",
    "p99":          "^",
}

# One colour per variable
C_RTT  = "#378ADD"
C_FPS  = "#E07B39"
C_SENT = "#1D9E75"
C_RECV = "#9B59B6"

# ── load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

# ── derive FPS per participant-trial session ────────────────────────────────────────
session_keys = ["benchmark_id", "participant_id", "trial", "participants"]
fps_df = (
    df.groupby(session_keys)
    .agg(
        n_steps=("timestep", "count"),
        t_min=("action_sent_time", "min"),
        t_max=("action_sent_time", "max"),
    )
    .reset_index()
)
fps_df["elapsed"] = fps_df["t_max"] - fps_df["t_min"]
fps_df = fps_df[fps_df["elapsed"] > 0].copy()
fps_df["fps"] = fps_df["n_steps"] / fps_df["elapsed"]

# ── helper: percentile table grouped by participants ──────────────────────────
def percentile_by_participants(source_df, value_col):
    rows = []
    for p, label in zip(PERCENTILES, PERCENTILE_LABELS):
        grp = (
            source_df.groupby("participants")[value_col]
            .quantile(p / 100)
            .reset_index()
        )
        grp.columns = ["participants", "value"]
        grp["percentile"] = label
        rows.append(grp)
    return pd.concat(rows, ignore_index=True)

rtt_pct  = percentile_by_participants(df,     "rtt_ms")
fps_pct  = percentile_by_participants(fps_df, "fps")
sent_pct = percentile_by_participants(df,     "bytes_sent")
recv_pct = percentile_by_participants(df,     "bytes_received")

# actual participant values for x-ticks
x_ticks = sorted(df["participants"].unique())

# ── plot helpers ──────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.0)

def draw_lines(ax, data, color):
    """Draw one line per percentile, all in the same colour."""
    for label in PERCENTILE_LABELS:
        sub = data[data["percentile"] == label].sort_values("participants")
        ax.plot(
            sub["participants"], sub["value"],
            color=color,
            linestyle=LINESTYLES[label],
            marker=MARKERS[label],
            markersize=6,
            linewidth=1.8,
        )

def finish_ax(ax, ylabel, title=None):
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks, fontsize=9)
    ax.set_xlabel("Participants", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(axis="y", labelsize=9)
    if title:
        ax.set_title(title, fontsize=11, pad=8)
    sns.despine(ax=ax)

# ── figure ────────────────────────────────────────────────────────────────────
fig, (ax_rtt, ax_fps, ax_tp) = plt.subplots(1, 3, figsize=(15, 5))

# RTT
draw_lines(ax_rtt, rtt_pct, C_RTT)
finish_ax(ax_rtt, "RTT (ms)", "Round-trip time")

# FPS
draw_lines(ax_fps, fps_pct, C_FPS)
finish_ax(ax_fps, "FPS", "Frames per second")

# Throughput — dual y-axis
ax_tp2 = ax_tp.twinx()
draw_lines(ax_tp,  sent_pct, C_SENT)
draw_lines(ax_tp2, recv_pct, C_RECV)

ax_tp.set_xticks(x_ticks)
ax_tp.set_xticklabels(x_ticks, fontsize=9)
ax_tp.set_xlabel("Participants", fontsize=10)
ax_tp.set_ylabel("Bytes sent",     fontsize=10, color=C_SENT)
ax_tp2.set_ylabel("Bytes received", fontsize=10, color=C_RECV)
ax_tp.tick_params(axis="y", labelcolor=C_SENT, labelsize=9)
ax_tp2.tick_params(axis="y", labelcolor=C_RECV, labelsize=9)
ax_tp.set_title("Throughput", fontsize=11, pad=8)
sns.despine(ax=ax_tp, right=False)

# ── shared legends ────────────────────────────────────────────────────────────
# Legend 1: line style = percentile  (shown once, on first axis)
style_handles = [
    mlines.Line2D([], [], color="grey", linestyle=LINESTYLES[l],
                  marker=MARKERS[l], markersize=6, label=l)
    for l in PERCENTILE_LABELS
]
ax_rtt.legend(handles=style_handles, title="Percentile",
              fontsize=9, title_fontsize=9, loc="best")

# Legend 2: colour = variable  (shown on throughput axis)
var_handles = [
    mlines.Line2D([], [], color=C_SENT, linestyle="-", label="bytes sent"),
    mlines.Line2D([], [], color=C_RECV, linestyle="-", label="bytes received"),
]
ax_tp.legend(handles=var_handles, title="Direction",
             fontsize=9, title_fontsize=9, loc="best")

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_PATH}")
plt.show()
