# Drainage blockage detection — mmWave FMCW radar readings

This repository holds raw radar captures used in research on **detecting blockages in drainage systems** with **frequency-modulated continuous-wave (FMCW)** **millimeter-wave (mmWave)** radar.

## Hardware and software


| Item               | Details                                                         |
| ------------------ | --------------------------------------------------------------- |
| Radar-on-chip      | [Texas Instruments IWR1843](https://www.ti.com/product/IWR1843) |
| Data capture       | DCA1000 (raw ADC stream to host)                                |
| Configuration tool | TI **mmWave Studio 2.1.1.0**                                    |


Binary captures are stored as `adc_data.bin` (Git LFS; see [.gitattributes](.gitattributes)).

## Dataset layout

| Top-level folder            | Description                                                                                      |
| --------------------------- | ------------------------------------------------------------------------------------------------ |
| `Empty Pipe`                | Dry / empty pipe; trials numbered `1` … `8`                                                      |
| `Water Only (Baseline)`      | Water-filled pipe baseline; **distance from pipe bottom** `10cm`, `15cm`, `20cm`, `25cm`; numbered trials under each |
| `Pebbles And Soil with Water` | Blockage-like material (pebbles and soil) with water; same distance-from-bottom folders and trial structure          |
| `OtherDebris with Water`    | Other debris(coconut husks, plastic bags, leaves) with water; same distance-from-bottom folders and trial structure  |


Each leaf folder contains a single raw capture:

```text
Empty Pipe/<trial>/adc_data.bin
<Condition>/<distance>/<trial>/adc_data.bin
```

The second pattern applies to `Water Only(Baseline)`, `PebblesAndSoil with Water`, and `OtherDebris with Water`. The `distance` folder is one of `10cm`, `15cm`, `20cm`, `25cm` and denotes **distance from the bottom of the pipe** to the sensor for that capture.

**Note that the blockage material is always at `10cm` from the bottom of the pipe.**

Example paths:

- `Empty Pipe/3/adc_data.bin`
- `Water Only(Baseline)/20cm/5/adc_data.bin`

## IWR1843 sensor configuration (mmWave Studio)

Values below are taken directly from the **SensorConfig** tab in mmWave Studio 2.1.1.0 used for collection. Radar System: **Single Chip**, capture card: **DCA1000**.

### Profile (Profile Id 0)


| Parameter                  | Value      |
| -------------------------- | ---------- |
| Start Freq (GHz)           | 77.000000  |
| Frequency Slope (MHz/µs)   | 65.998     |
| Idle Time (µs)             | 100.00     |
| TX Start Time (µs)         | 0.00       |
| ADC Start Time (µs)        | 6.00       |
| ADC Samples                | 512        |
| Sample Rate (ksps)         | 10000      |
| Ramp End Time (µs)         | 60.00      |
| RX Gain (dB)               | 28         |
| RF Gain Target             | 30 dB      |
| VCO Select                 | VCO1       |
| HPF1 Corner Freq           | 175K       |
| HPF2 Corner Freq           | 350K       |
| O/p Pwr Backoff TX0 (dB)   | 0          |
| O/p Pwr Backoff TX1 (dB)   | 0          |
| O/p Pwr Backoff TX2 (dB)   | 0          |
| Phase Shifter TX0 (deg)    | 0.000      |
| Phase Shifter TX1 (deg)    | 0.000      |
| Phase Shifter TX2 (deg)    | 0.000      |
| Config Bandwidth (MHz)     | 3959.88    |
| Effective Bandwidth (MHz)  | 3379.1     |


### Inter Rx Gain Phase Freq Control


| Parameter         | Rx0  | Rx1  | Rx2  | Rx3  |
| ----------------- | ---- | ---- | ---- | ---- |
| Dig Gain (dB)     | 0.0  | 0.0  | 0.0  | 0.0  |
| Dig Ph Shift (Deg)| 0.00 | 0.00 | 0.00 | 0.00 |

ProfileIndex: **0**

### Chirp


| Parameter                       | Value    |
| ------------------------------- | -------- |
| Profile Id                      | 0        |
| Start Chirp for Cfg             | 0        |
| End Chirp for Cfg               | 0        |
| Start Freq Var (MHz)            | 0.000000 |
| Frequency Slope Var (MHz/µs)    | 0.000    |
| Idle Time Var (µs)              | 0.00     |
| ADC Start Var (µs)              | 0.00     |
| TX Enable for current chirp     | TX0      |


### Frame


| Parameter              | Value            |
| ---------------------- | ---------------- |
| Start Chirp TX         | 0                |
| End Chirp TX           | 0                |
| No of Chirp Loops      | 128              |
| No of Frames           | 100              |
| Dummy Chirps (End)     | 0                |
| Periodicity (ms)       | 100.000000       |
| Trigger Delay (µs)     | 0.00             |
| Active-Ramp Duty Cycle | 7.7 %            |
| Duty Cycle             | 20.5 %           |
| Trigger Select         | SoftwareTrigger  |


### Enable Dynamic Power Save in Inter-chirp

TX, RX, and LO Dist enabled.

## Analysis scripts

The [`code/`](code) folder contains Python scripts used to parse the raw captures and produce the figures and tables referenced in the analysis.

| Script | Purpose |
| ------ | ------- |
| [`plot_empty_pipe_profiles.py`](code/plot_empty_pipe_profiles.py) | Range profiles for the empty-pipe baseline trials |
| [`plot_water_only_profiles.py`](code/plot_water_only_profiles.py) | Range profiles for the water-only baseline across distances |
| [`plot_range_profile_comparison.py`](code/plot_range_profile_comparison.py) | Side-by-side range-profile comparison across pipe conditions |
| [`plot_mti_range_doppler_maps.py`](code/plot_mti_range_doppler_maps.py) | Range–Doppler maps with MTI clutter suppression |
| [`plot_aoa_spectra_bin9.py`](code/plot_aoa_spectra_bin9.py) | Angle-of-arrival spectra at range bin 9 |
| [`plot_aoa_spectra_water_surface.py`](code/plot_aoa_spectra_water_surface.py) | Angle-of-arrival spectra at the water-surface range bin |
| [`plot_doppler_energy_bar_chart.py`](code/plot_doppler_energy_bar_chart.py) | Bar chart of Doppler-band energy by condition |
| [`print_doppler_energy_table.py`](code/print_doppler_energy_table.py) | Prints the Doppler-band energy summary table |

## License and citation

If you use this dataset, please cite the associated research publication when it is available, and retain any license terms added by the project authors.