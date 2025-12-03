# RTP Voice Network Simulator

An educational tool for simulating RTP packet transmission with network impairments and dejitter buffering.

## Features

- **Audio Input**: Upload WAV, MP3, OGG, FLAC, or M4A files
- **RTP Packetization**: Simulates RTP packet creation with timestamps and sequence numbers
- **Network Impairments**:
  - Delay and jitter simulation
  - Multiple loss models (Random, Gilbert-Elliott, Burst)
  - Packet reordering
- **Receiver Processing**:
  - Dejitter buffer with configurable playout delay
  - Loss concealment (zero, repeat, interpolate)
- **Quality Assessment**:
  - Objective metrics (MSE, SNR, PESQ)
  - Audio waveform comparison
  - Packet timeline visualization
- **Educational Focus**: Step-by-step simulation with explanations

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rtp-voice-simulator