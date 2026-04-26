import re
import gc
import warnings
import numpy as np
from pathlib import Path
from collections import defaultdict

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
BIN_40   = 9
VEL_BAND = 0.5
DEPTHS   = [10, 15, 20, 25]

READINGS_DIR = Path(r"D:\New folder\Readings")


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


def print_doppler_energy_table_at_40cm():
    print("\n── Table 2: Doppler energy at bin 9 (≈ 39.95 cm) ──")
    results = defaultdict(list)

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
                    rfft, nc = read_bin(fpath)
                    power, vel_axis = apply_mti_filter_and_doppler_fft(rfft, nc)
                    del rfft; gc.collect()
                    vel_mask = (np.abs(vel_axis) <= VEL_BAND) & (np.abs(vel_axis) > 1e-9)
                    lo = max(0, BIN_40 - 1); hi = min(N_BINS, BIN_40 + 2)
                    e  = float(10 * np.log10(np.mean(power[vel_mask, lo:hi]) + 1e-30))
                    results[(mc, depth)].append(round(e, 1))
                except Exception as ex:
                    print(f"  !! {fpath.name}: {ex}")

    print(f"\n  {'Fill Depth':<12} {'Water Only':>12} {'PebblesAndSoil':>16} {'OtherDebris':>13}")
    print("  " + "-" * 56)
    for depth in DEPTHS:
        w  = results.get(("water_only",   depth), [])
        ps = results.get(("pebbles_soil", depth), [])
        od = results.get(("other_debris", depth), [])
        print(f"  {depth:>2} cm       "
              f"  {np.mean(w):>10.1f} dB"
              f"  {np.mean(ps):>14.1f} dB"
              f"  {np.mean(od):>11.1f} dB")


if __name__ == "__main__":
    print_doppler_energy_table_at_40cm()
