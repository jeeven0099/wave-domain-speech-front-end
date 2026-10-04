# Result Definitions

## Primary comparison

`results/audiomnist/audiomnist_100hz_main_results.csv` contains the paired
AudioMNIST results used in the primary table.

- `Baseline`: 16 kHz waveform to 40-bin log-Mel features at 100 frames/s.
- `Our Work`: five resonator channels represented by 100 Hz AC-RMS envelopes.
- `preprocessing_median_ms`: feature extraction and normalization only.
- `nn_median_ms`: batch-1 neural-network forward pass only.
- `compute_median_ms`: sum of the two stage medians.
- `dynamic_energy_mj`: `0.30 W * compute_median_ms`.
- `nn_macs`: profiled neural-network MACs for one one-second input.

The 0.30 W coefficient is the measured incremental Raspberry Pi draw used in
the experiment. Energy values exclude audio playback, microphone bias,
amplification, DAQ/ADC power, and host idle power.

## Supporting comparisons

- `ablation_100hz_pi_results.csv` compares the 100 Hz resonator models with
  their earlier 400 Hz versions.
- `icassp_results_table.csv` retains the 16 kHz and 400 Hz paired benchmark used
  in the envelope-rate analysis.
- `results/raspberry_pi/benchmark_summary_19system.csv` contains the complete
  19-system Raspberry Pi benchmark summary.
- `results/raspberry_pi/raw_timings.csv` contains the individual measured runs.

## Interpretation boundary

The current captures begin with high-rate five-channel DAQ waveforms. The
software AC-RMS extraction is included in measured preprocessing latency. A
system in which analog envelope extraction precedes five low-rate ADC channels
would have a different acquisition and power profile and is not measured here.
