import gc
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

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
SENSOR_HEIGHT_CM = 44.7

READINGS_DIR = Path(r"D:\New folder\Readings")
OUT_BIAS     = Path(r"D:\New folder\output\range_bias_investigation")
OUT_BIAS.mkdir(parents=True, exist_ok=True)


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


def plot_empty_pipe_individual_range_profiles():
    print("\n── Figure B: Empty pipe individual profiles ──")
    empty_dir = READINGS_DIR / "Empty Pipe"
    mask = RANGE_AXIS_CM <= RANGE_MAX_CM

    profiles, peak_bins = [], []
    for rec_dir in sorted(empty_dir.iterdir()):
        fpath = next(rec_dir.rglob("*.bin"), None)
        if fpath is None:
            continue
        rfft, _ = read_bin(fpath)
        prof = range_profile(rfft)
        del rfft; gc.collect()
        profiles.append(to_dbfs(prof))
        peak_bins.append(int(np.argmax(prof)))

    fig, axes = plt.subplots(4, 2, figsize=(12, 18))
    for i, (prof, peak) in enumerate(zip(profiles, peak_bins)):
        ax = axes[i // 2][i % 2]
        ax.plot(RANGE_AXIS_CM[mask], prof[mask], color="#1f77b4", lw=1.2)
        obs_x, obs_y = RANGE_AXIS_CM[peak], prof[peak]
        ax.axvline(obs_x, color="red", ls="--", lw=1, alpha=0.8, label="Observed peak")
        ax.axvline(SENSOR_HEIGHT_CM, color="green", ls=":", lw=1, alpha=0.7, label="Expected (pipe bottom)")
        ax.text(obs_x + 0.5, obs_y + 0.5, f"({obs_x:.1f} cm, {obs_y:.1f} dBFS)",
                fontsize=7, color="darkred", ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="none"), zorder=6)
        for b in [peak - 1, peak, peak + 1]:
            if 0 <= b < N_BINS:
                ax.axvline(b * RANGE_RES_CM, color="gray", ls=":", lw=0.4, alpha=0.5)
        ax.set_xlabel("Distance from sensor (cm)", fontsize=9)
        ax.set_ylabel("Magnitude (dBFS)", fontsize=9)
        ax.set_title(f"Recording #{i + 1}", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Figure B — Empty Pipe: All 8 Individual Range Profiles", fontsize=11, y=1.02)
    fig.tight_layout()
    out = OUT_BIAS / "F1_empty_pipe_all_8_profiles.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


if __name__ == "__main__":
    plot_empty_pipe_individual_range_profiles()
