import re
import gc
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

warnings.filterwarnings("ignore")

NUM_ADC = 512
NUM_RX  = 4
FS      = 10e6
SLOPE   = 65.998e12
C       = 3e8
FC      = 77e9
LAM     = C / FC
T_ADC   = NUM_ADC / FS
B       = SLOPE * T_ADC
N_BINS  = NUM_ADC // 2
DPI     = 150
VEL_BAND = 0.5

VIZ_COLORS = {
    "Empty Pipe":     "#555555",
    "Water":          "#1f77b4",
    "PebblesAndSoil": "#8c6d31",
    "OtherDebris":    "#d62728",
}

READINGS_DIR = Path(r"D:\New folder\Readings")
OUT_ROOT     = Path(r"D:\New folder\output")
OUT_ROOT.mkdir(parents=True, exist_ok=True)


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


def apply_mti_filter_and_doppler_fft(rfft, nc):
    v_res    = LAM / (2 * nc * T_ADC)
    vel_axis = (np.arange(nc) - nc // 2) * v_res
    rfft_mti = rfft - rfft.mean(axis=0, keepdims=True)
    dop_fft  = np.fft.fftshift(np.fft.fft(rfft_mti, axis=0), axes=0)
    power    = np.mean(np.abs(dop_fft)**2, axis=1)
    del rfft_mti, dop_fft
    return power, vel_axis


def compute_mean_range_profile(rfft):
    return np.mean(np.abs(rfft), axis=(0, 1))


def plot_doppler_energy_bar_chart_all_scenarios():
    print("\n── Figure 4b (bar): Doppler energy comparison ──")

    scenarios = []
    for mat_dir in sorted(READINGS_DIR.iterdir()):
        if not mat_dir.is_dir(): continue
        material = mat_dir.name
        subdirs  = [d for d in mat_dir.iterdir() if d.is_dir()]
        has_dist = any(re.search(r"\d+cm", d.name, re.I) for d in subdirs)
        if has_dist:
            for dist_dir in sorted(subdirs, key=lambda d: int(re.search(r"\d+", d.name).group())):
                if not re.search(r"\d+cm", dist_dir.name, re.I): continue
                files = sorted(dist_dir.rglob("*.bin"))
                if files:
                    scenarios.append({"label": f"{material}_{dist_dir.name}",
                                      "files": files, "material": material})
        else:
            files = sorted(mat_dir.rglob("*.bin"))
            if files:
                scenarios.append({"label": material, "files": files, "material": material})

    labels, energies, colors = [], [], []
    for sc in scenarios:
        all_profs = []
        for fpath in sc["files"]:
            try:
                rfft, nc = read_bin(fpath)
                prof_lin = compute_mean_range_profile(rfft)
                power, vel_axis = apply_mti_filter_and_doppler_fft(rfft, nc)
                all_profs.append((prof_lin, power, vel_axis))
                del rfft, power; gc.collect()
            except Exception as e:
                print(f"  !! {fpath.name}: {e}")
        if not all_profs: continue

        mean_prof = np.mean([p for p, _, _ in all_profs], axis=0)
        peak_bin  = int(np.argmax(mean_prof))
        lo        = max(0, peak_bin - 1); hi = min(N_BINS, peak_bin + 2)
        vel       = all_profs[0][2]
        vel_mask  = (np.abs(vel) <= VEL_BAND) & (np.abs(vel) > 1e-9)
        e_vals    = [10 * np.log10(np.mean(pw[vel_mask, lo:hi]) + 1e-30) for _, pw, _ in all_profs]
        energies.append(float(np.mean(e_vals)))
        labels.append(sc["label"].replace("_", "\n"))
        colors.append(VIZ_COLORS.get(sc["material"], "gray"))

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(labels)), energies, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Doppler energy at peak range bin (dB)", fontsize=11)
    ax.set_title("Figure 4b — Doppler Energy Comparison\n"
                 "Water Only vs blockage materials across all fill depths", fontsize=12)
    ax.grid(True, axis="y", alpha=0.4)
    handles = [Patch(color=c, label=m) for m, c in VIZ_COLORS.items()]
    ax.legend(handles=handles, fontsize=9, loc="upper right")
    plt.tight_layout()
    out = OUT_ROOT / "doppler_energy_comparison.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


if __name__ == "__main__":
    plot_doppler_energy_bar_chart_all_scenarios()
