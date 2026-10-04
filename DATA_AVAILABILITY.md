# Data Availability

## Included

- Aggregate AudioMNIST accuracy, latency, MAC, parameter, and energy results.
- Individual Raspberry Pi timing measurements.
- COMSOL-derived five-resonator subset rankings and summary metrics.
- Selected publication figures and the scripts used to generate them.

## Not included

- AudioMNIST WAV files. Obtain these from the official AudioMNIST distribution.
- Raw five-channel DAQ captures.
- COMSOL project files and full complex transfer-function exports.
- Trained PyTorch checkpoints.
- Generated feature caches and playback stimuli.

## Expected local inputs

The AudioMNIST training and benchmark scripts expect:

```text
AudioMNIST WAV root
fixed speaker-split metadata
five-channel DAQ capture root
trained checkpoint directories
```

The Raspberry Pi scripts expose the relevant paths through their configuration
section and environment variables. The bundle verifier documents the required
layout without distributing the underlying data or model weights.
