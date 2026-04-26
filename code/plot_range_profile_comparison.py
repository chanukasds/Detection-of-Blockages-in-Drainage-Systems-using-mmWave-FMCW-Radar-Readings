import re
import gc
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")

NUM_ADC       = 512
NUM_RX        = 4
FS            = 10e6
SLOPE         = 65.998e12
C             = 3e8
FC            = 77e9
LAM           = C / FC
T_ADC         = NUM_ADC / FS
B             = SLOPE * T_ADC
RANGE_RES_CM  = (C / (2 * B)) * 100
N_BINS        = NUM_ADC // 2
RANGE_AXIS_CM = np.arange(N_BINS) * RANGE_RES_CM
DBFS_REF      = NUM_ADC * (2**15 - 1)
DPI           = 150
RANGE_MAX_CM  = 60.0
BIN_40        = 9
DEPTHS    = [10, 15, 20, 25]
WATER_BIN = {10: 9, 15: 8, 20: 7, 25: 6}

MATERIAL_COLORS = {
    "water_only":   "#1f77b4",
    "pebbles_soil": "#8B4513",
    "other_debris": "#d62728",
}
MATERIAL_LABELS = {
    "water_only":   "Water Only",
    "pebbles_soil": "PebblesAndSoil + Water",
    "other_debris": "OtherDebris + Water",
}

READINGS_DIR = Path(r"D:\New folder\Readings")
OUT_SUMMARY  = Path(r"D:\New folder\output\blockage_detection_analysis")
OUT_SUMMARY.mkdir(parents=True, exist_ok=True)


def to_dbfs(linear):
    return 20.0 * np.log10(np.maximum(linear, 1e-12) / DBFS_REF)


def read_bin(path):
    raw  = np.fromfile(str(path), dtype=np.int16)
    n    = len(raw)
    nc   = n // (2 * NUM_ADC * NUM_RX)
    lvds = np.empty(n // 2, dtype=np.complex64)
    lvds[0::2] = raw[0::4].astype(np.float32) + 1j * raw[2::4].astype(np.float32)
    lvds[1::2] = raw[1::4].astype(np.float32) + 1j * raw[3::4].astype(np.float32)
    del raw
    cube = lvds[:nc * NUM_RX * NUM_ADC].reshape(nc, NUM_RX, NUM_ADC).copy()
    del lvds
    rfft = np.fft.fft(cube, axis=2)[:, :, :N_BINS].astype(np.complex64)
    del cube
    return rfft, nc


def range_profile(rfft):
    return np.mean(np.abs(rfft), axis=(0, 1))


def classify(path_str):
    p = path_str.lower()
    if "pebbles" in p or "soil" in p: return "pebbles_soil"
    if "debris" in p:                  return "other_debris"
    if "water"  in p:                  return "water_only"
    return None


def plot_range_profile_comparison_all_materials_by_depth():
    print("\n── Figure 1: Range profile comparison by depth ──")

    profiles = defaultdict(list)
    for mat_dir in sorted(READINGS_DIR.iterdir()):
        if not mat_dir.is_dir(): continue
        mc = classify(str(mat_dir))
        if mc is None: continue
        for depth_dir in sorted(mat_dir.iterdir()):
            if not depth_dir.is_dir(): continue
            m = re.search(r"(\d+)cm", depth_dir.name, re.I)
            if not m: continue
            depth = int(m.group(1))
            if depth not in DEPTHS: continue
            for rec_dir in sorted(depth_dir.iterdir()):
                if not rec_dir.is_dir(): continue
                fpath = next(rec_dir.rglob("*.bin"), None)
                if fpath is None: continue
                try:
                    rfft, _ = read_bin(fpath)
                    profiles[(mc, depth)].append(range_profile(rfft))
                    del rfft; gc.collect()
                except Exception as e:
                    print(f"  !! {fpath.name}: {e}")

    mask = RANGE_AXIS_CM <= RANGE_MAX_CM
    rax  = RANGE_AXIS_CM[mask]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.subplots_adjust(hspace=0.38, wspace=0.28, left=0.07, right=0.97, top=0.90, bottom=0.07)

    for idx, depth in enumerate(DEPTHS):
        ax       = axes[idx // 2][idx % 2]
        surf_bin = WATER_BIN[depth]
        surf_cm  = RANGE_AXIS_CM[surf_bin]

        for mc in ["water_only", "pebbles_soil", "other_debris"]:
            recs = profiles.get((mc, depth), [])
            if not recs: continue
            mean_lin = np.stack(recs, axis=0).mean(axis=0)
            mean_db  = to_dbfs(mean_lin)
            ax.plot(rax, mean_db[mask], color=MATERIAL_COLORS[mc], lw=2.0, label=MATERIAL_LABELS[mc])

            if depth in [15, 20, 25]:
                ref_idx = int(np.argmin(np.abs(rax - 40.0)))
                ref_cm  = float(rax[ref_idx])
                ref_db  = float(mean_db[mask][ref_idx])
                y_off   = {"water_only": +5.5, "pebbles_soil": 0.0, "other_debris": -5.5}
                ax.annotate(
                    f"@40cm: {ref_db:.1f} dBFS",
                    xy=(ref_cm, ref_db),
                    xytext=(ref_cm + 1.5, ref_db + y_off.get(mc, 0.0)),
                    fontsize=6.0, color=MATERIAL_COLORS[mc], ha="left", va="center",
                    arrowprops=dict(arrowstyle="->", color=MATERIAL_COLORS[mc], lw=0.6, shrinkA=2, shrinkB=2),
                )

        ax.axvline(surf_cm, color="green", ls="--", lw=1.0, alpha=0.7,
                   label=f"Surface bin {surf_bin} ({surf_cm:.1f} cm)")
        if depth in [15, 20, 25]:
            ax.axvline(RANGE_AXIS_CM[BIN_40], color="grey", ls=":", lw=0.8, alpha=0.6,
                       label=f"40 cm ref (bin {BIN_40})")
        ax.set_xlim(0, RANGE_MAX_CM)
        ax.set_xlabel("Distance from sensor (cm)", fontsize=9)
        ax.set_ylabel("Magnitude (dBFS)", fontsize=9)
        ax.set_title(f"Fill Depth {depth} cm — Surface bin {surf_bin} ({surf_cm:.1f} cm)", fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.25)

    fig.suptitle("Figure 1 — Range Profile Comparison by Fill Depth\n"
                 "Water Only vs PebblesAndSoil vs OtherDebris (mean across recordings)",
                 fontsize=12, y=0.97)
    out = OUT_SUMMARY / "S1_range_profile_comparison_by_depth.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


if __name__ == "__main__":
    plot_range_profile_comparison_all_materials_by_depth()
