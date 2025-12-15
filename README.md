# RTP Voice Network Simulator

## Overview

The **RTP Voice Network Simulator** is an **educational and experimental application** designed to explore **Quality of Service (QoS)** effects in real-time voice communication. It simulates the end-to-end transmission of audio over **RTP (Real-time Transport Protocol)** while exposing how network impairments such as **delay, jitter, packet loss, and reordering** impact audio quality.

The primary goal of this project is **understanding and visualization**, rather than building a production-grade media system. By making network behavior observable and configurable, the simulator helps bridge the gap between **QoS theory and practical behavior**.

---

## Key Objectives

* Demonstrate how real-time audio is packetized and transmitted using RTP
* Visualize the effects of network impairments on voice quality
* Compare different packet loss models, including **realistic burst-loss behavior**
* Study the role of **dejitter buffers** and playout delay
* Provide an interactive platform for experimentation and learning

---

## Main Features

### Audio Input

* Supports common audio formats: **WAV, MP3, OGG, FLAC, M4A**
* Audio is treated as a continuous real-time signal rather than a static file

### RTP Packetization

* Simulation of RTP packet creation
* Each packet includes:

  * Sequence number
  * Timestamp
* Packet timing reflects real-time transmission behavior

### Network Impairment Models

The simulator supports multiple network impairment mechanisms to reflect different QoS conditions:

* **Delay and Jitter**: Variable packet transmission delays
* **Packet Loss Models**:

  * Random (Bernoulli) loss
  * **Gilbert–Elliott model** for realistic bursty loss
  * Explicit burst loss patterns
* **Packet Reordering**: Simulates out-of-order delivery

These models allow comparison between simplistic and realistic network assumptions.

### Receiver-Side Processing

* **Dejitter buffer** with configurable playout delay
* Packet reordering handling
* **Loss concealment strategies**:

  * Zero insertion
  * Packet repetition
  * Interpolation

### Quality Assessment and Visualization

* Objective quality metrics:

  * Mean Squared Error (MSE)
  * Signal-to-Noise Ratio (SNR)
  * PESQ (where supported)
* Visual comparison of original and received audio waveforms
* Packet timeline and state visualization

---

## Educational Focus

This simulator is intentionally designed as a **step-by-step system**:

* Each stage of the RTP pipeline is exposed
* QoS effects are not hidden by aggressive correction mechanisms
* Users can observe cause–effect relationships directly

The application is well suited for:

* Networking and multimedia courses
* QoS and QoE experimentation
* Demonstrations of RTP behavior under non-ideal networks

---

## Architecture Overview

The application follows a modular structure:

```
Audio Input
    ↓
RTP Packetization
    ↓
Network Impairment Simulation
    ↓
Receiver & Dejitter Buffer
    ↓
Quality Evaluation & Visualization
```

The simulation logic is decoupled from the user interface, enabling clean experimentation and future extensions.

---

## Technology Stack

* **Python**
* **Streamlit** for interactive visualization and control
* Scientific and audio-processing libraries from the Python ecosystem

Streamlit is used as a lightweight UI layer, enabling rapid experimentation without frontend complexity.

---

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd rtp-voice-simulator
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
streamlit run app.py
```

---

## License

This project is released under a **permissive open-source license** to encourage reuse, experimentation, and reproducibility in academic and research contexts.

---

## Limitations and Scope

* This is **not** a production VoIP system
* Focuses on **QoS observation**, not optimization
* Real-time guarantees are simulated, not enforced

These limitations are intentional and align with the educational goals of the project.

---

## Future Extensions

Possible future improvements include:

* Multi-state or adaptive network models
* QoE-driven adaptation strategies
* Advanced concealment techniques
* Integration with live RTP streams

---

## Summary

The RTP Voice Network Simulator provides a clear, interactive, and realistic environment for studying **QoS effects in real-time audio communication**, with particular emphasis on **packet loss burstiness, jitter, and buffering behavior**.
