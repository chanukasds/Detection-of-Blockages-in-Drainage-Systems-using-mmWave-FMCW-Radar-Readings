import re
import gc
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
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
VEL_BAND      = 0.5
VEL_DISPLAY   = 2.0
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
OUT_RDMAP    = Path(r"D:\New folder\output\doppler_mti_maps")
OUT_RDMAP.mkdir(parents=True, exist_ok=True)


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


def classify(path_str):
    p = path_str.lower()
    if "pebbles" in p or "soil" in p: return "pebbles_soil"
    if "debris" in p:                  return "other_debris"
    if "water"  in p:                  return "water_only"
    return None


def load_and_average_mti_doppler_maps_per_material_and_depth():
    data = {}
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
            maps_lin, vel_ref = [], None
            for rec_dir in sorted(depth_dir.iterdir()):
                if not rec_dir.is_dir(): continue
                fpath = next(rec_dir.rglob("*.bin"), None)
                if fpath is None: continue
                try:
                    rfft, nc = read_bin(fpath)
                    power, vel_axis = apply_mti_filter_and_doppler_fft(rfft, nc)
                    power_dB = 10 * np.log10(power + 1e-30)
                    maps_lin.append(10 ** (power_dB / 10))
                    if vel_ref is None:
                        vel_ref = vel_axis
                    del rfft, power
                    gc.collect()
                except Exception as e:
                    print(f"  !! {fpath.name}: {e}")
            if maps_lin and vel_ref is not None:
                avg_dB = 10 * np.log10(np.mean(maps_lin, axis=0) + 1e-30)
                data[(mc, depth)] = (vel_ref, avg_dB, len(maps_lin))
                print(f"  Loaded {MATERIAL_LABELS[mc]:30s} {depth}cm  n={len(maps_lin)}")
    return data


def render_single_material_doppler_heatmap(ax, vel, rdmap_dB, mc, depth, vmin, vmax, show_xlabel=True, show_ylabel=True):
    rm  = RANGE_AXIS_CM <= RANGE_MAX_CM
    vm  = np.abs(vel) <= VEL_DISPLAY
    pcm = ax.pcolormesh(RANGE_AXIS_CM[rm], vel[vm], rdmap_dB[np.ix_(vm, rm)],
                        cmap="inferno", vmin=vmin, vmax=vmax, shading="auto")
    ref_cm = RANGE_AXIS_CM[WATER_BIN[depth]]
    ax.axhline( VEL_BAND, color="cyan",  ls="--", lw=1.0, alpha=0.85)
    ax.axhline(-VEL_BAND, color="cyan",  ls="--", lw=1.0, alpha=0.85, label=f"±{VEL_BAND} m/s")
    ax.axhline(0,         color="white", ls=":",  lw=0.8, alpha=0.5)
    ax.axvline(ref_cm,    color="lime",  ls="--", lw=1.0, alpha=0.75, label=f"Surface {ref_cm:.1f} cm")
    ax.set_xlim(0, RANGE_MAX_CM)
    ax.set_ylim(-VEL_DISPLAY, VEL_DISPLAY)
    ax.set_title(MATERIAL_LABELS[mc], fontsize=10, color=MATERIAL_COLORS[mc], fontweight="bold")
    if show_xlabel: ax.set_xlabel("Range from sensor (cm)", fontsize=9)
    if show_ylabel: ax.set_ylabel("Radial velocity (m/s)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    return pcm


def plot_mti_range_doppler_maps_per_depth():
    print("\n── Figures 4b–4e: MTI Range-Doppler maps ──")
    data       = load_and_average_mti_doppler_maps_per_material_and_depth()
    fig_labels = {10: "4b", 15: "4c", 20: "4d", 25: "4e"}
    row_pairs  = [
        ("water_only", "pebbles_soil"),
        ("water_only", "other_debris"),
    ]

    for depth in DEPTHS:
        fig = plt.figure(figsize=(14, 10))
        gs  = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.05],
                                hspace=0.36, wspace=0.28,
                                left=0.07, right=0.93, top=0.88, bottom=0.08)
        axes = [[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(2)]
        cax  = fig.add_subplot(gs[:, 2])

        all_vals = []
        for mc in ["water_only", "pebbles_soil", "other_debris"]:
            entry = data.get((mc, depth))
            if entry is None: continue
            vel, rdmap_dB, _ = entry
            vm = np.abs(vel) <= VEL_DISPLAY
            rm = RANGE_AXIS_CM <= RANGE_MAX_CM
            all_vals.append(rdmap_dB[np.ix_(vm, rm)])

        if not all_vals:
            plt.close(fig); continue

        combined = np.concatenate([v.ravel() for v in all_vals])
        vmax     = np.percentile(combined, 99)
        vmin     = vmax - 50

        pcm = None
        for r, pair in enumerate(row_pairs):
            for c, mc in enumerate(pair):
                ax    = axes[r][c]
                entry = data.get((mc, depth))
                if entry is None:
                    ax.set_visible(False); continue
                vel, rdmap_dB, n = entry
                pcm = render_single_material_doppler_heatmap(
                    ax, vel, rdmap_dB, mc, depth, vmin, vmax,
                    show_xlabel=(r == 1), show_ylabel=(c == 0))
                ax.text(0.97, 0.97, f"n={n}", transform=ax.transAxes,
                        ha="right", va="top", fontsize=8, color="white",
                        bbox=dict(fc="black", alpha=0.4, pad=2))

        if pcm is not None:
            cbar = fig.colorbar(pcm, cax=cax)
            cbar.set_label("Doppler power (dB)", fontsize=9)
            cbar.ax.tick_params(labelsize=8)

        ref_cm = RANGE_AXIS_CM[WATER_BIN[depth]]
        fig.suptitle(
            f"Figure {fig_labels[depth]} — MTI Range-Doppler Maps: Fill Depth {depth} cm\n"
            f"Row 1: Water Only vs PebblesAndSoil   |   Row 2: Water Only vs OtherDebris\n"
            f"Surface bin {WATER_BIN[depth]} = {ref_cm:.1f} cm  |  "
            f"Cyan dashed: ±{VEL_BAND} m/s  |  Lime dashed: water-surface range",
            fontsize=10, y=0.98)

        out = OUT_RDMAP / f"rdmap_mti_{depth}cm.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")


if __name__ == "__main__":
    plot_mti_range_doppler_maps_per_depth()
