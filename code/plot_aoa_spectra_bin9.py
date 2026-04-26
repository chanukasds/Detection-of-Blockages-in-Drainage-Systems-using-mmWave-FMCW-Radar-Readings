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
DPI           = 150
BIN_40        = 9
DEPTHS    = [10, 15, 20, 25]
WATER_BIN = {10: 9, 15: 8, 20: 7, 25: 6}
AoA_ANGLES_DEG = np.arange(-90, 91, dtype=float)

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
OUT_AOA      = Path(r"D:\New folder\output\aoa_analysis")
OUT_AOA.mkdir(parents=True, exist_ok=True)


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


def classify(path_str):
    p = path_str.lower()
    if "pebbles" in p or "soil" in p: return "pebbles_soil"
    if "debris" in p:                  return "other_debris"
    if "water"  in p:                  return "water_only"
    return None


def compute_aoa(snapshot, angles=AoA_ANGLES_DEG):
    rx = np.arange(NUM_RX)
    return np.array([
        abs(np.dot(np.exp(-1j * np.pi * rx * np.sin(np.deg2rad(a))).conj(), snapshot))**2
        for a in angles
    ])


def beamwidth(P_norm):
    above = AoA_ANGLES_DEG[P_norm >= 0.5]
    return float(above[-1] - above[0]) if len(above) >= 2 else 180.0


def plot_aoa_bartlett_spectra_at_fixed_blockage_bin():
    print("\n── Figure AoA3c: AoA spectra at bin 9 (40 cm) only ──")

    spectra = defaultdict(list)
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
                    lo   = max(0, BIN_40 - 1); hi = min(N_BINS, BIN_40 + 2)
                    snap = rfft.mean(axis=0)[:, lo:hi].mean(axis=1)
                    P    = compute_aoa(snap)
                    mx   = P.max()
                    spectra[(mc, depth)].append(P / mx if mx > 0 else P)
                    del rfft; gc.collect()
                except Exception as e:
                    print(f"  !! {fpath.name}: {e}")

    bin9_cm = RANGE_AXIS_CM[BIN_40]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.subplots_adjust(hspace=0.40, wspace=0.28, left=0.07, right=0.97, top=0.88, bottom=0.07)

    for idx, depth in enumerate(DEPTHS):
        ax       = axes[idx // 2][idx % 2]
        surf_bin = WATER_BIN[depth]
        surf_cm  = RANGE_AXIS_CM[surf_bin]
        bw_vals  = {}

        for mc in ["water_only", "pebbles_soil", "other_debris"]:
            recs = spectra.get((mc, depth), [])
            if not recs: continue
            mean_P = np.array(recs).mean(axis=0)
            bw     = beamwidth(mean_P)
            bw_vals[mc] = bw
            n = len(recs)
            ax.plot(AoA_ANGLES_DEG, mean_P, color=MATERIAL_COLORS[mc], lw=2.2,
                    label=f"{MATERIAL_LABELS[mc]}  BW={bw:.0f}°  (n={n})")

        w_bw      = bw_vals.get("water_only")
        delta_str = ""
        if w_bw:
            for mc2, lbl2 in [("pebbles_soil", "P&S"), ("other_debris", "Debris")]:
                if mc2 in bw_vals:
                    delta_str += f"  ΔBW({lbl2})={bw_vals[mc2] - w_bw:+.0f}°"

        ax.axhline(0.5, color="grey", ls="--", lw=1.0, alpha=0.7, label="-3 dB threshold")
        ax.axvline(0,   color="grey", ls=":",  lw=0.8, alpha=0.6)
        ax.set_xlim(-90, 90)
        ax.set_ylim(-0.05, 1.15)
        ax.set_xticks(range(-90, 91, 15))
        ax.set_xlabel("Angle (°)", fontsize=9)
        ax.set_ylabel("Normalised power", fontsize=9)
        ax.set_title(
            f"Fill depth {depth} cm  |  Bin 9 = {bin9_cm:.1f} cm (blockage position)\n"
            f"Water-surface bin = {surf_bin} ({surf_cm:.1f} cm){delta_str}",
            fontsize=9)
        ax.legend(fontsize=7.5, loc="upper right")
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Figure AoA3c — AoA Bartlett Spectra at Bin 9 (≈39.95 cm) — Fixed Blockage Reference\n"
        "All three material classes at each fill depth, normalised to peak\n"
        "ΔBW = blockage beamwidth minus Water Only beamwidth at same depth",
        fontsize=11, y=0.97)
    out = OUT_AOA / "4c_aoa_spectra_bin9_only.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


if __name__ == "__main__":
    plot_aoa_bartlett_spectra_at_fixed_blockage_bin()
