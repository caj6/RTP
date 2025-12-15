"""
Main Streamlit application for RTP Voice Network Simulator.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import sys
import os
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # Importation of modules from correct directories
    from audio.processor import AudioProcessor
    from audio.codec import PCMCodec
    from rtp.packet import RTPPacket  
    from rtp.packetizer import Packetizer 
    from rtp.loss_models import LossModelFactory 
    from network.simulator import NetworkSimulator
    from network.receiver import Receiver
    from visualization.plots import Visualization
    from visualization.metrics import QualityMetrics

    imports_ok = True

except ImportError as e:
    st.error(f"Import error: {e}")
    st.info(
        """
    **Troubleshooting Steps:**
    1. Make sure all module files are in correct directories
    2. Check that each directory has an `__init__.py` file
    3. Run from the RFIP directory: `streamlit run app.py`
    4. Install missing packages: `pip install -r requirements.txt`
    """
    )

    # Show directory structure help
    with st.expander("📁 Expected Directory Structure"):
        st.code(
            """
RTP/
├── audio/
│   ├── __init__.py
│   ├── codec.py
│   └── processor.py
├── rtp/
│   ├── __init__.py
│   ├── loss_models.py
│   ├── packet.py
│   └── packetizer.py
├── network/
│   ├── __init__.py
│   ├── receiver.py
│   └── simulator.py
├── visualization/
│   ├── __init__.py
│   ├── metrics.py
│   └── plots.py
├── app.py
├── README.md
├── requirements.txt
└── test_audio.py
        """
        )

    imports_ok = False


def app():
    """Main application entry point."""
    if 'generated_audio' not in st.session_state:
        st.session_state.generated_audio = None

    # Page configuration
    st.set_page_config(
        page_title="RTP Voice Network Simulator",
        page_icon="🎧",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Title and description
    st.title("🎧 RTP Voice Network Simulator")
    st.markdown(
        """
    An educational tool to simulate RTP packet transmission over impaired networks.
    Upload audio, configure network conditions, and observe effects on voice quality.
    
    **⚠️ Important Notes:**
    - Use WAV files only (other formats not supported)
    - Recommended audio length: 3-10 seconds
    - Results are simulated for educational purposes
    """
    )

    if not imports_ok:
        st.stop()

    # Initialize components
    audio_processor = AudioProcessor()
    packetizer = Packetizer()
    network_sim = NetworkSimulator()
    receiver = Receiver()
    visualizer = Visualization()
    metrics = QualityMetrics()

    # Sidebar configuration
    with st.sidebar:
        st.header("🎛️ Configuration")

        # Audio input
        st.subheader("📁 Audio Input")
        audio_file = st.file_uploader(
            "Upload Audio File",
            type=["wav"],
            help="""
            **📊 Format Requirements:**
            - **Format:** WAV audio files only
            - **Duration:** 3-10 seconds recommended
              - Too short (<1s): PESQ evaluation unreliable
              - Too long (>30s): Slow simulation
            - **Sample Rate:** Any rate (will be resampled)
            - **Channels:** Mono or stereo (auto-converted to mono)
            
            **⚙️ Processing Pipeline:**
            1. **Channel Conversion:** Stereo → Mono (if needed)
            2. **Resampling:** To target sample rate below
            3. **Normalization:** Prevents clipping
            4. **Packetization:** Based on frame size
            
            **🎯 Educational Purpose:**
            Upload voice clips to understand how network conditions affect VoIP quality.
            """,
        )

        # Processing settings
        st.subheader("⚙️ Audio Processing")

        target_sample_rate = st.selectbox(
            "Target Sample Rate",
            [8000, 16000, 22050, 44100],
            index=1,
            help="""
            **🎵 Impact on Audio Quality & Simulation:**
            
            **Important:** Changing this will resample your input audio!
            
            **8 kHz (Telephony Standard):**
            - Frequency range: 300-3400 Hz
            - Small packets (64 kbps for PCM)
            - Standard for traditional VoIP
            - **PESQ optimized** for this rate (narrowband)
            - **Impact:** High frequencies removed, voice optimized
            
            **16 kHz (Wideband VoIP):**
            - Frequency range: 50-7000 Hz
            - Better voice clarity, more natural sound
            - Modern VoIP standard (Skype, Teams, Zoom)
            - **PESQ wideband mode** available
            - **Impact:** Minimal quality loss, preserves voice details
            
            **22.05 kHz / 44.1 kHz (Music Quality):**
            - Not typical for VoIP systems
            - Larger packets, more bandwidth required
            - **⚠️ PESQ NOT VALID** for these rates
            - **Impact:** May cause downsampling artifacts
            - **Educational use only** - shows effect of bandwidth
            
            **📊 Resampling Effects on Simulation:**
            - Higher → Lower: Loses high frequencies (low-pass filtering)
            - Lower → Higher: No quality gain (empty high frequencies)
            - Resampling changes packet size: Higher rate = more data per packet
            - Aliasing possible if original has frequencies > Nyquist limit
            
            **⚠️ PESQ Limitation:**
            PESQ only valid for 8kHz (narrowband) and 16kHz (wideband).
            Other rates will show "N/A" or use auto-resampled version.
            
            **Recommendation for evaluation:** Use 8k or 16k for accurate PESQ scores.
            """,
        )

        # Warning for non-standard VoIP rates
        if target_sample_rate not in [8000, 16000]:
            st.warning(
                f"""
            ⚠️ **Non-standard VoIP Rate Selected: {target_sample_rate} Hz**
            
            - Not typical for real VoIP systems
            - PESQ scores may be inaccurate or unavailable
            - Results for educational demonstration only
            - Consider using 8kHz or 16kHz for valid PESQ evaluation
            """,
                icon="⚠️",
            )

        frame_size_ms = st.slider(
            "Packet Frame Size (ms)",
            10,
            60,
            20,
            5,
            help="""
            **📦 Audio Duration per RTP Packet:**
            
            **Technical Details:**
            - Each packet contains N milliseconds of audio
            - Frame size = samples per packet = Sample Rate × Frame Size (ms) ÷ 1000
            
            **Trade-offs:**
            
            **Small frames (10ms):**
            - ✅ More responsive to voice activity changes
            - ✅ Better jitter tolerance (smaller time slices)
            - ✅ Lower latency for interactive conversation
            - ❌ More overhead (40-byte RTP header per packet)
            - ❌ More sensitive to individual packet loss
            - ❌ Higher packet rate = more processing load
            
            **Large frames (60ms):**
            - ✅ Less overhead (fewer headers)
            - ✅ More efficient for bandwidth
            - ✅ Better tolerance to random loss patterns
            - ❌ More audio lost per dropped packet (bigger gaps)
            - ❌ Higher latency (must wait for more audio to accumulate)
            - ❌ Less responsive to voice/silence transitions
            
            **Industry Standards:**
            - **G.711/G.729:** 20ms typical
            - **Opus:** Variable (2.5-60ms), adaptive
            - **Mobile VoIP:** 20-40ms
            - **Real-time gaming voice:** 10-20ms
            
            **Formula:** 
            Samples per packet = Sample Rate × Frame Size (ms) ÷ 1000
            Packet rate = 1000 ÷ Frame Size (packets/second)
            
            **Simulation impact:** Affects packet count, loss impact granularity, and latency.
            """,
        )

        # Calculate and show frame size info
        samples_per_packet = int(target_sample_rate * frame_size_ms / 1000)
        packet_rate = 1000 / frame_size_ms
        st.caption(
            f"**Technical:** {samples_per_packet} samples/packet = {samples_per_packet * 2:,} bytes (PCM16) = {packet_rate:.1f} packets/sec"
        )

        # Network settings
        st.subheader("🌐 Network Impairments")

        # Delay configuration
        st.markdown("#### ⏱️ Delay & Jitter Configuration")

        base_delay_ms = st.slider(
            "Base Network Delay (ms)",
            0,
            500,
            50,
            10,
            help="""
            **Constant Propagation Delay:**
            
            **What it represents:**
            - Physical distance (≈1ms per 200km in fiber optic)
            - Router/switch processing (1-10ms per hop)
            - Serialization delay (time to put bits on wire)
            - Propagation through cables and equipment
            
            **Typical values for different networks:**
            - **LAN (Local Area Network):** < 10ms
            - **Metropolitan Area:** 10-30ms
            - **Cross-country (USA/EU):** 50-100ms
            - **International (trans-oceanic):** 100-300ms
            - **Satellite link:** 500ms+ (geostationary)
            - **Low Earth Orbit Satellite:** 20-40ms
            
            **Effect on conversation quality:**
            - **< 150ms:** Good for interactive conversation
            - **150-300ms:** Noticeable but acceptable with echo control
            - **> 300ms:** Difficult for interactive talk, turn-taking issues
            - **> 500ms:** Very difficult, often requires push-to-talk
            
            **ITU-T G.114 Recommendations:**
            - 0-150ms: Acceptable for most applications
            - 150-400ms: Acceptable with awareness
            - >400ms: Unacceptable for general network planning
            
            **Educational insight:** This is one-way delay. Round-trip delay is ~2× this value.
            """,
        )

        delay_distribution = st.selectbox(
            "Delay Distribution Model",
            ["Constant + Normal", "Exponential", "Pareto"],
            index=0,
            help="""
            **📈 Statistical Models for Delay Variation (Jitter):**
            
            **Constant + Normal:**
            - Base delay + Normally distributed jitter
            - Jitter parameter = standard deviation (σ) in ms
            - Most common for well-behaved, stable networks
            - Mathematically: delay = base + N(0, jitter²)
            - Assumes independent delays, Gaussian distribution
            - Good for wired networks with steady traffic
            
            **Exponential:**
            - Models queuing delays in routers/buffers
            - Memoryless property: next delay independent of previous
            - Mathematically: delay = base + Exponential(mean=jitter)
            - More realistic for congested networks with random arrivals
            - Right-skewed distribution (more chance of moderate delays)
            - Common in networks with Poisson packet arrivals
            
            **Pareto (Heavy-Tailed):**
            - Models extreme delay events and burstiness
            - "Internet traffic is more bursty than Poisson"
            - Long tail distribution (rare but very large delays possible)
            - Mathematically: Pareto(shape=2.5, scale=jitter)
            - Models self-similar network traffic patterns
            - More realistic for modern internet with multimedia traffic
            
            **🎓 Educational Insight:**
            Real networks show all three behaviors at different times:
            - **Normal:** Baseline wired connections
            - **Exponential:** Moderate congestion periods  
            - **Pareto:** Network overload, routing changes, wireless issues
            
            **Simulation difference:** Affects delay variability pattern and buffer requirements.
            """,
        )

        jitter_ms = st.slider(
            "Network Jitter (ms)",
            0,
            200,
            30,
            5,
            help="""
            **🔄 Delay Variation (Standard Deviation for Normal model):**
            
            **What is jitter?**
            - Variation in packet arrival times
            - Measured as standard deviation of delays
            - Caused by queuing in routers, variable paths, load balancing
            
            **Sources in real networks:**
            - **Network congestion:** Queuing in routers during peak times
            - **Variable routing paths:** BGP changes, traffic engineering
            - **Load balancing:** Packets taking different paths
            - **Wireless interference:** WiFi channel contention
            - **Cross-traffic:** Other applications sharing bandwidth
            - **Bufferbloat:** Large buffers causing excessive queuing delays
            
            **Impact on VoIP quality:**
            
            **Low (< 20ms):**
            - Minimal impact on quality
            - Small dejitter buffer sufficient (20-40ms)
            - Typical of well-managed enterprise networks
            
            **Medium (20-50ms):**
            - Needs dejitter buffer (50-100ms)
            - Some audio artifacts may be noticeable
            - Common in residential broadband during peak hours
            
            **High (50-100ms):**
            - Requires large buffers (100-200ms)
            - Significant audio quality impact
            - Common in congested networks, public WiFi
            
            **Very High (> 100ms):**
            - Difficult to compensate with buffers
            - Often requires adaptive jitter buffers
            - Common in satellite, mobile, or heavily congested networks
            
            **Receiver buffer size calculation:**
            Buffer size ≈ 2 × jitter + base delay
            
            **📊 Jitter measurement standards:**
            - RFC 3550: Interarrival jitter formula
            - Typical internet jitter: 10-50ms
            - Target for good VoIP: < 20ms
            
            **Simulation note:** This parameter affects how spread out delays are around the mean.
            """,
        )

        # Show distribution preview
        if jitter_ms > 0:
            with st.expander("📊 Distribution Preview"):
                if delay_distribution == "Constant + Normal":
                    st.markdown(
                        f"""
                    **Normal Distribution (Gaussian):**
                    - Mean delay = {base_delay_ms} ms
                    - Standard deviation (σ) = {jitter_ms} ms
                    - **68% of packets:** {base_delay_ms - jitter_ms:.0f} to {base_delay_ms + jitter_ms:.0f} ms
                    - **95% of packets:** {base_delay_ms - 2*jitter_ms:.0f} to {base_delay_ms + 2*jitter_ms:.0f} ms
                    - **99.7% of packets:** {base_delay_ms - 3*jitter_ms:.0f} to {base_delay_ms + 3*jitter_ms:.0f} ms
                    - **Shape:** Bell curve, symmetric
                    """
                    )
                elif delay_distribution == "Exponential":
                    st.markdown(
                        f"""
                    **Exponential Distribution:**
                    - Base delay = {base_delay_ms} ms
                    - Mean additional delay = {jitter_ms} ms (rate λ = {1/jitter_ms:.3f})
                    - **Memoryless property:** Future independent of past
                    - **63% of packets:** Delay < {base_delay_ms + jitter_ms:.0f} ms
                    - **86% of packets:** Delay < {base_delay_ms + 2*jitter_ms:.0f} ms
                    - **95% of packets:** Delay < {base_delay_ms + 3*jitter_ms:.0f} ms
                    - **Shape:** Right-skewed, many small delays, few large delays
                    """
                    )
                elif delay_distribution == "Pareto":
                    st.markdown(
                        f"""
                    **Pareto Distribution (Heavy-Tailed):**
                    - Base delay = {base_delay_ms} ms
                    - Scale parameter = {jitter_ms} ms
                    - Shape parameter = 2.5
                    - **Heavy-tailed:** Occasional very large delays
                    - **80/20 rule:** 80% of extra delay from 20% of packets
                    - **Self-similar:** Looks similar at different time scales
                    - **Bursty delay patterns:** Clusters of high delays
                    - **Shape:** Power-law distribution, long right tail
                    """
                    )

        # Loss model
        st.subheader("📉 Packet Loss Models")
        loss_model_type = st.selectbox(
            "Select Loss Model",
            ["None", "Random", "Gilbert-Elliott"],
            index=1,
            help="""
            **🎲 Statistical Models of Packet Loss:**
            
            **None (Perfect Network):**
            - No packet loss occurs
            - Baseline for comparison
            - Unrealistic but useful for debugging and understanding other impairments
            - Shows effect of delay/jitter without loss
            
            **Random (Bernoulli) Model:**
            - Each packet independently lost with probability p
            - Models random bit errors, transmission errors
            - Memoryless property: Loss events independent
            - Formula: P(loss) = constant for every packet
            - Simple but not realistic for congested networks
            - Good for modeling physical layer errors
            
            **Gilbert-Elliott (2-State Markov):**
            - Two states: Good (G) and Bad (B) states
            - State transitions: G⇄B with transition probabilities
            - Different loss rates in each state (low in good, high in bad)
            - Models **bursty loss** - packets lost in clusters
            - More realistic for congested networks with traffic bursts
            - Parameters: p_gb (G→B), p_bg (B→G), loss_good, loss_bad
            
            **🎓 Research Insight:**
            Internet packet loss is bursty, not random.
            Studies show loss bursts of 2-10 packets common.
            Gilbert-Elliott captures this bursty behavior well.
            
            **Simulation choice:**
            - **Random:** Simple, predictable, good for baseline
            - **Gilbert-Elliott:** Realistic, shows burst effects, good for buffer tuning
            """,
        )

        loss_model_config = {}
        if loss_model_type == "Random":
            loss_rate = st.slider(
                "Random Loss Probability",
                0.0,
                0.5,
                0.05,
                0.01,
                format="%.3f",
                help="""
                **🎯 Independent Loss Probability:**
                
                **Quality Impact Guidelines (ITU-T G.113):**
                - **< 1%:** Excellent VoIP quality (MOS 4.0-4.5)
                - **1-3%:** Good quality, some artifacts (MOS 3.5-4.0)
                - **3-5%:** Fair quality, noticeable gaps (MOS 3.0-3.5)
                - **5-10%:** Poor quality, difficult conversation (MOS 2.5-3.0)
                - **> 10%:** Very poor, often unusable (MOS < 2.5)
                
                **Real-world examples by network type:**
                - **Fiber optic networks:** < 0.1% (virtually lossless)
                - **Enterprise wired LAN:** 0.1-1% (well-managed)
                - **Residential broadband:** 1-3% (during peak hours)
                - **Mobile 4G/5G networks:** 1-5% (variable)
                - **Congested/old networks:** 5-20% (poor conditions)
                - **Satellite links:** 1-10% (weather dependent)
                
                **PESQ Correlation (approximate):**
                - 0% loss ≈ PESQ 4.5 (perfect)
                - 1% loss ≈ PESQ 3.8-4.0 (good)
                - 3% loss ≈ PESQ 3.2-3.5 (fair)
                - 5% loss ≈ PESQ 2.8-3.2 (poor)
                - 10% loss ≈ PESQ 2.3-2.8 (bad)
                
                **📊 Network planning guidelines:**
                - VoIP target: < 1% end-to-end
                - Enterprise SLA: < 0.5%
                - Mobile target: < 3%
                - Emergency services: < 0.1%
                
                **Simulation tip:** Start with 5% to see noticeable effects, then adjust.
                """,
            )
            loss_model_config = {"type": "random", "loss_rate": loss_rate}

            # Show loss preview
            expected_loss = int(loss_rate * 100)
            st.caption(
                f"**Expected loss:** ~{expected_loss}% of packets ({expected_loss} out of 100)"
            )

        elif loss_model_type == "Gilbert-Elliott":
            st.markdown("##### Gilbert-Elliott Parameters")
            st.info(
                """
            **📊 Two-State Markov Model (Bursty Loss):**
            
            **States:**
            - **Good State (G):** Low loss probability (typical: 0-0.5%)
            - **Bad State (B):** High loss probability (typical: 30-80%)
            
            **Transitions:**
            - **P(Good→Bad):** p_gb = probability of entering lossy state
            - **P(Bad→Good):** p_bg = probability of recovering to good state
            
            **Steady-state analysis:**
            - % time in Bad state = p_gb / (p_gb + p_bg)
            - Average loss rate = (%Bad × loss_bad) + (%Good × loss_good)
            - Average burst length (in Bad state) = 1 / p_bg packets
            - Average good period length = 1 / p_gb packets
            
            **Real-world interpretation:**
            - Good state: Normal network operation
            - Bad state: Congestion episode, routing change, interference burst
            """,
                icon="ℹ️",
            )

            col1, col2 = st.columns(2)
            with col1:
                p_gb = st.slider(
                    "P(Good → Bad)",
                    0.0,
                    0.2,
                    0.02,
                    0.01,
                    format="%.3f",
                    help="**Probability of entering lossy state**\n\nTypical values:\n- Stable networks: 0.01-0.05\n- Variable networks: 0.05-0.10\n- Unstable networks: 0.10-0.20\n\nHigher = more frequent bursts",
                )
                loss_good = st.slider(
                    "Loss in Good State (%)",
                    0.0,
                    0.5,
                    0.1,
                    0.05,
                    format="%.2f",
                    help="**Very low loss in good state**\n\nTypical values:\n- Excellent: 0-0.1%\n- Good: 0.1-0.3%\n- Acceptable: 0.3-0.5%\n\nRepresents baseline loss during normal operation",
                )
                loss_good = loss_good / 100  # Convert percentage to probability

            with col2:
                p_bg = st.slider(
                    "P(Bad → Good)",
                    0.0,
                    1.0,
                    0.3,
                    0.01,
                    format="%.3f",
                    help="**Probability of recovering to good state**\n\nTypical values:\n- Fast recovery: 0.5-1.0\n- Moderate recovery: 0.2-0.5\n- Slow recovery: 0.05-0.2\n\nHigher = shorter loss bursts",
                )
                loss_bad = st.slider(
                    "Loss in Bad State (%)",
                    0.0,
                    100.0,
                    60.0,
                    5.0,
                    format="%.1f",
                    help="**High loss in bad state**\n\nTypical values:\n- Moderate congestion: 30-50%\n- Severe congestion: 50-80%\n- Complete outage: 80-100%\n\nRepresents loss during congestion episodes",
                )
                loss_bad = loss_bad / 100  # Convert percentage to probability

            # Calculate steady-state statistics
            if p_gb + p_bg > 0:
                p_bad_steady = p_gb / (p_gb + p_bg)
                p_good_steady = p_bg / (p_gb + p_bg)
                avg_loss_rate = (p_bad_steady * loss_bad) + (p_good_steady * loss_good)
                avg_burst_length = 1 / p_bg if p_bg > 0 else float("inf")
                avg_good_length = 1 / p_gb if p_gb > 0 else float("inf")

                st.caption(
                    f"""
                **Steady-state analysis:**
                - Time in Good state: {p_good_steady:.1%}
                - Time in Bad state: {p_bad_steady:.1%}
                - Average loss rate: {avg_loss_rate:.2%}
                - Average burst length: {avg_burst_length:.1f} packets
                - Average good period: {avg_good_length:.1f} packets
                - Burstiness index: {avg_loss_rate/(loss_bad - loss_good):.2f}
                """
                )

            loss_model_config = {
                "type": "gilbert_elliott",
                "p_gb": p_gb,
                "p_bg": p_bg,
                "loss_good": loss_good,
                "loss_bad": loss_bad,
            }
        else:
            loss_model_config = {"type": "none"}

        reorder_prob = st.slider(
            "Packet Reordering Probability",
            0.0,
            0.3,
            0.02,
            0.01,
            format="%.3f",
            help="""
            **🔄 Out-of-Order Arrival Probability:**
            
            **Causes in real networks:**
            - **Multi-path routing:** Packets take different paths with different delays
            - **Load balancing:** Across multiple links or routes
            - **Router queue dynamics:** Different queuing delays in routers
            - **Route flapping:** Rapid routing protocol changes
            - **Wireless retransmissions:** MAC layer retries (not typical for RTP)
            
            **Impact on VoIP:**
            - Receiver must reorder using RTP sequence numbers
            - Late reordered packets become late arrivals (effectively lost)
            - Increases effective jitter (reordering adds to delay variation)
            - Consumes buffer space while waiting for missing packets
            
            **Typical values by network type:**
            - **LAN / Data center:** < 1% (virtually none)
            - **Enterprise WAN:** 1-3% (some reordering)
            - **Internet backbone:** 3-8% (moderate reordering)
            - **Wireless/mobile:** 5-15% (significant reordering)
            - **Multi-homed paths:** 10-25% (high reordering)
            
            **Receiver handling:**
            1. Sort packets by sequence number
            2. Hold packets in buffer waiting for missing ones
            3. Discard packets that arrive too late
            4. Adjust playout buffer based on reordering pattern
            
            **RFC 4737 defines reordering metrics:**
            - Reordering extent: How far packets are displaced
            - Reordering frequency: How often it occurs
            - Reordering-free runs: Gaps between reordered packets
            
            **Simulation note:** Reordering creates gaps that need to be filled with concealment.
            """,
        )

        # Receiver settings
        st.subheader("🎚️ Receiver Settings")

        playout_delay_ms = st.slider(
            "Initial Playout Delay (ms)",
            0,
            500,
            100,
            10,
            help="""
            **⏰ Receiver Buffering Delay (Dejitter Buffer):**
            
            **Purpose:**
            - Compensate for network jitter and delay variation
            - Allow time for packet reordering and late arrivals
            - Smooth out variable network delays for consistent playback
            - Convert network delay variation into constant playout delay
            
            **Trade-off (Quality vs Latency):**
            - **Too low:** Many late packets → poor quality (clicks, gaps)
            - **Too high:** Excessive latency → conversation lag, echo issues
            - **Optimal:** Balanced based on network conditions
            
            **Rule of thumb calculations:**
            - Minimum buffer = 2 × jitter + base_delay
            - Conservative buffer = 4 × jitter + base_delay
            - Typical VoIP applications: 60-200ms
            - Adaptive jitter buffers: 20-400ms (dynamic adjustment)
            
            **Buffer operation:**
            1. Packets arrive at variable times
            2. Held in buffer until playout time
            3. Playout time = Arrival time + playout delay
            4. Late packets (arriving after playout time) are discarded
            
            **⚠️ Important:** This adds to total end-to-end delay!
            Total latency = Network delay + Playout delay
            
            **Adaptive buffer strategies:**
            - **Fixed:** Constant delay (simplest)
            - **Adaptive:** Adjusts based on measured jitter
            - **Queue-based:** Adjusts based on buffer fill level
            - **Loss-based:** Adjusts based on packet loss rate
            
            **Recommendation:** Start with 100ms, adjust based on results.
            """,
        )

        max_buffer_ms = st.slider(
            "Maximum Buffer Tolerance (ms)",
            0,
            1000,
            200,
            10,
            help="""
            **⏳ Late Packet Threshold (Buffering Window):**
            
            **Interpretation:** "Accept packets arriving up to N ms after their scheduled playout time"
            
            **Detailed operation:**
            1. Packet scheduled for playout at time P
            2. Packet actually arrives at time A
            3. Calculate lateness = A - P
            4. Decision:
               - If lateness ≤ max_buffer_ms: Play packet (possibly late but within tolerance)
               - If lateness > max_buffer_ms: Discard as "too late"
            
            **Alternative interpretation:** "Extend buffer window to N ms beyond scheduled playout"
            
            **Effect:** Converts excessive network delay into packet loss
            Trade-off: Late playback vs. complete loss
            
            **Settings guidelines:**
            - **Conservative (strict):** 0-50ms beyond playout time
              - Few late packets, more loss
              - Lower latency, more gaps
            - **Moderate:** 50-200ms beyond playout time  
              - Balance between loss and delay
              - Typical for VoIP applications
            - **Aggressive (tolerant):** 200-500ms beyond playout time
              - Fewer lost packets, more late playback
              - Higher latency, smoother playback
            - **Adaptive:** Varies based on network conditions
            
            **Buffer vs. Max Buffer relationship:**
            - **Playout delay:** When to START playing packets
            - **Max buffer tolerance:** How LONG to wait for late packets
            - Total waiting window = Playout delay + Max buffer tolerance
            
            **Example:** 
            Playout delay = 100ms, Max buffer = 200ms
            - Packet scheduled at t=100ms
            - Accept if arrives by t=300ms (100 + 200)
            - Discard if arrives after t=300ms
            
            **Trade-off analysis:**
            - Higher max_buffer: Fewer late packets discarded, more latency
            - Lower max_buffer: More packets discarded as late, less latency
            """,
        )

        # Show buffer relationship
        total_latency = base_delay_ms + playout_delay_ms
        max_wait_time = playout_delay_ms + max_buffer_ms
        st.caption(
            f"""
        **Latency Analysis:**
        - Network delay: {base_delay_ms} ms
        + Playout buffer: {playout_delay_ms} ms
        = **Total latency: {total_latency} ms**
        
        **Buffer Window:**
        - Schedule playout: {playout_delay_ms} ms after arrival
        - Accept packets up to: {max_wait_time} ms after arrival
        - Late threshold: {max_buffer_ms} ms beyond scheduled time
        """
        )

        concealment = st.selectbox(
            "Packet Loss Concealment",
            ["zero", "repeat", "interpolate"],
            help="""
            **🔧 Techniques to Handle Missing Packets (PLC):**
            
            **Zero-filling (Silence substitution):**
            - Insert silence/zeros for lost packets
            - Simplest implementation, lowest complexity
            - Creates audible gaps, clicks, and discontinuities
            - May cause voice clipping and unnatural pauses
            - Used in basic systems, not recommended for VoIP
            
            **Frame repetition (Last packet repeat):**
            - Repeat last successfully received packet
            - Smoother than silence, maintains some continuity
            - May cause tonal artifacts, buzzing sounds
            - Can create "stuttering" effect during long losses
            - Standard technique in G.711 Appendix I
            - Good for short losses (< 60ms)
            
            **Interpolation (Waveform substitution):**
            - Repeat with volume fade (attenuation)
            - Cross-fade between previous and next packets
            - Smoother transitions, more natural sound
            - Higher computational cost
            - Used in advanced codecs (Opus, G.722.2/AMR-WB)
            - Can handle longer losses better
            
            **🎓 Advanced PLC techniques:**
            - **Waveform replication:** Pattern matching in history buffer
            - **Pitch-based repetition:** Repeat at pitch period boundaries
            - **Neural network PLC:** AI-based prediction (Zoom, Teams)
            - **Forward Error Correction:** Send redundant data
            - **Codec-specific PLC:** Built into modern codecs
            
            **ITU-T Standards:**
            - G.711 Appendix I: Packet loss concealment
            - G.722 Annex C: PLC for 7 kHz audio
            - G.729 Annex B: Voice activity detection and comfort noise
            
            **Recommendation:** Use 'interpolate' for best quality, 'repeat' for balance.
            """,
        )

        # Run button
        st.markdown("---")
        st.markdown("### 🚀 Run Simulation")

        # Validation checks
        validation_ok = True
        if audio_file is None:
            st.warning("Please upload an audio file first")
            validation_ok = False

        # PESQ validation warnings
        if target_sample_rate not in [8000, 16000]:
            st.warning(
                f"⚠️ **PESQ Validation:** Sample rate {target_sample_rate} Hz not valid for PESQ. "
                f"Use 8kHz or 16kHz for accurate perceptual quality evaluation."
            )

        # Latency warnings
        if total_latency > 300:
            st.warning(
                f"⚠️ **High Latency Warning:** Total latency ({total_latency}ms) exceeds 300ms. "
                f"May cause conversation difficulties (ITU-T G.114 limit)."
            )
        elif total_latency > 150:
            st.info(
                f"ℹ️ **Moderate Latency:** Total latency ({total_latency}ms) may be noticeable "
                f"in conversation but acceptable with echo control."
            )

        run_button = st.button(
            "▶️ Start Simulation",
            type="primary",
            use_container_width=True,
            disabled=not validation_ok,
        )

    # Main content
    if audio_file is None and st.session_state.generated_audio is None:
        # Welcome screen
        display_welcome_screen()
        return

    # Check if we have generated audio in session state
    if st.session_state.generated_audio is not None and audio_file is None:
        audio_file = io.BytesIO(st.session_state.generated_audio)

    # Display loaded audio
    try:
        st.subheader("🎵 Original Audio")

        # Show audio info
        with st.spinner("Loading audio info..."):
            audio_file.seek(0)
            audio_data, sample_rate = audio_processor.load_audio(audio_file)
            duration = len(audio_data) / sample_rate

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Original Rate", f"{sample_rate:,} Hz")
            with col2:
                st.metric("Duration", f"{duration:.2f} s")
            with col3:
                channels = (
                    "Mono"
                    if audio_data.ndim == 1
                    else f"Stereo ({audio_data.shape[1]} ch)"
                )
                st.metric("Channels", channels)
            with col4:
                st.metric("Samples", f"{len(audio_data):,}")

        # Duration validation
        if duration < 1.0:
            st.error(
                "⚠️ **PESQ Validation Error:** Audio too short for reliable PESQ evaluation (minimum 1 second required)"
            )
        elif duration < 3.0:
            st.warning(
                "ℹ️ **Short Audio:** For best PESQ results, use 3+ seconds of audio"
            )
        elif duration > 30:
            st.warning("Audio longer than 30 seconds may cause slow simulation")

        # Display audio player
        audio_file.seek(0)  # Reset file pointer
        st.audio(audio_file, format="audio/wav")

        # Show waveform preview
        with st.expander("📊 Original Waveform Preview"):
            try:
                audio_file.seek(0)
                audio_data, sr = audio_processor.load_audio(audio_file)
                audio_data = audio_processor._to_mono(audio_data)

                # Create simple waveform plot
                fig, ax = plt.subplots(figsize=(10, 3))
                time_axis = np.arange(len(audio_data)) / sr

                # Downsample if too many points
                if len(time_axis) > 5000:
                    step = len(time_axis) // 5000
                    time_axis = time_axis[::step]
                    audio_data = audio_data[::step]

                ax.plot(
                    time_axis, audio_data, alpha=0.7, linewidth=0.5, color="#2E86AB"
                )
                ax.fill_between(time_axis, 0, audio_data, alpha=0.3, color="#2E86AB")
                ax.set_xlabel("Time (seconds)")
                ax.set_ylabel("Amplitude")
                ax.set_title("Original Audio Waveform")
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color="black", alpha=0.3, linewidth=0.5)
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not generate waveform: {e}")

        # Show run button
        if run_button:
            run_simulation(
                audio_file,
                audio_processor,
                packetizer,
                network_sim,
                receiver,
                visualizer,
                metrics,
                target_sample_rate,
                frame_size_ms,
                base_delay_ms,
                jitter_ms,
                loss_model_config,
                reorder_prob,
                playout_delay_ms,
                max_buffer_ms,
                concealment,
                delay_distribution,
            )

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info(
            """
        **Troubleshooting audio issues:**
        1. Ensure file is WAV format (not MP3 renamed to .wav)
        2. Try a shorter audio file (3-5 seconds)
        3. Check file isn't corrupted
        4. Try the test audio button below
        """
        )

        if st.button("🎵 Use Test Audio (1kHz tone, 1 second)"):
            if os.path.exists("test_audio.wav"):
                with open("test_audio.wav", "rb") as f:
                    audio_file = io.BytesIO(f.read())
                st.rerun()


def display_welcome_screen():
    """Display welcome screen when no audio is loaded."""
    st.info("👈 **Please upload a WAV audio file to begin simulation.**")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📋 Quick Start Guide")
        st.markdown(
            """
        1. **Upload** a WAV audio file (record a short voice clip)
        2. **Configure** network conditions in sidebar
        3. **Run** simulation to see effects
        4. **Analyze** results and learn
        
        """
        )

    with col2:
        st.markdown("### 🎓 Learning Objectives")
        st.markdown(
            """
        - **RTP Packetization**: How voice is divided into packets
        - **Network Impairments**: Effects of delay, jitter, loss
        - **Dejitter Buffering**: Receiver compensation techniques
        - **Quality Metrics**: Objective vs. perceptual quality
        - **Trade-offs**: Latency vs. quality decisions
        """
        )

    # Example scenarios
    st.markdown("---")
    st.markdown("### 💡 Example Scenarios to Explore")

    scenario_cols = st.columns(3)
    with scenario_cols[0]:
        st.markdown("**Good Network**")
        st.markdown(
            """
        - Delay: 50ms
        - Jitter: 20ms
        - Loss: 1% Random
        - Buffer: 100ms
        - **Expected PESQ:** ~4.0
        """
        )

    with scenario_cols[1]:
        st.markdown("**Moderate Network**")
        st.markdown(
            """
        - Delay: 100ms
        - Jitter: 50ms
        - Loss: 5% Random
        - Buffer: 200ms
        - **Expected PESQ:** ~3.2-3.5
        """
        )

    with scenario_cols[2]:
        st.markdown("**Poor Network**")
        st.markdown(
            """
        - Delay: 200ms
        - Jitter: 100ms
        - Loss: 15% Gilbert-Elliott
        - Buffer: 400ms
        - **Expected PESQ:** ~2.5-3.0
        """
        )

        # Show test file option
        if st.button("🎵 Use Test Audio File (1 sec 1kHz tone)"):
            with open("test_audio.wav", "rb") as f:
                audio_bytes = f.read()

            # Create download button and auto-load
            st.download_button(
                label="Download Test Audio",
                data=audio_bytes,
                file_name="test_audio.wav",
                mime="audio/wav",
            )
            st.info(
                "Download the test file, then upload it using the file uploader above."
            )

def generate_sine_wave(duration=3.0, frequency=1000, sample_rate=16000, amplitude=0.5):
    """
    Sine wave audio signal.
    Args:
        duration: Audio duration in seconds
        frequency: Sine wave frequency in Hz
        sample_rate: Sample rate in Hz
        amplitude: Amplitude (0.0 to 1.0)

    Returns:
        Audio data as numpy array
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * t)

    fade_samples = int(0.05 * sample_rate)  # 50ms fade
    if fade_samples > 0:
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        sine_wave[:fade_samples] *= fade_in
        sine_wave[-fade_samples:] *= fade_out

    return sine_wave


def run_simulation(
    audio_file,
    audio_processor,
    packetizer,
    network_sim,
    receiver,
    visualizer,
    metrics,
    target_sample_rate,
    frame_size_ms,
    base_delay_ms,
    jitter_ms,
    loss_model_config,
    reorder_prob,
    playout_delay_ms,
    max_buffer_ms,
    concealment,
    delay_distribution,
):
    """Run the complete simulation."""

    # Create progress tracker
    progress_bar = st.progress(0)
    status_text = st.empty()
    result_container = st.container()

    with result_container:
        st.header("📊 Simulation Results")

        # Step 1: Load and process audio
        status_text.text("📥 **Step 1/5: Loading and processing audio...**")
        audio_file.seek(0)
        audio_data, original_sample_rate = audio_processor.load_audio(audio_file)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Original Sample Rate", f"{original_sample_rate:,} Hz")

        processed_audio = audio_processor.process_audio(
            audio_data, original_sample_rate, target_sample_rate
        )

        with col2:
            st.metric("Processed Sample Rate", f"{target_sample_rate:,} Hz")

        duration = len(processed_audio) / target_sample_rate
        st.metric("Audio Duration", f"{duration:.2f} seconds")
        progress_bar.progress(20)

        # Validate audio length for PESQ
        if duration < 1.0:
            st.error(
                """
            ⚠️ **PESQ VALIDATION ERROR**
            
            **Audio Too Short:** PESQ requires minimum 1 second of audio.
            
            **Current duration:** {duration:.2f} seconds
            **Minimum required:** 1.0 second
            
            **Impact:** PESQ score will be unreliable or unavailable.
            
            **Recommendation:** Use longer audio clips (3-10 seconds recommended).
            """.format(
                    duration=duration
                )
            )
        elif duration < 3.0:
            st.warning(
                """
            ℹ️ **Short Audio Note**
            
            **Duration:** {duration:.2f} seconds
            
            PESQ evaluation works best with 3+ seconds of speech.
            Results may be less reliable with very short clips.
            """.format(
                    duration=duration
                )
            )

        # PESQ sample rate validation
        if target_sample_rate not in [8000, 16000]:
            st.error(
                """
            ⚠️ **PESQ VALIDATION ERROR**
            
            **Invalid Sample Rate:** PESQ scores are only valid for:
            - **8,000 Hz** (narrowband telephony)
            - **16,000 Hz** (wideband VoIP)
            
            **Current rate:** {target_sample_rate} Hz
            **Impact:** PESQ score will be inaccurate or unavailable.
            
            **Recommendation:** Use 8k or 16k for meaningful PESQ evaluation.
            """.format(
                    target_sample_rate=target_sample_rate
                )
            )

        # Step 2: Packetization
        status_text.text("📦 **Step 2/5: Packetizing audio...**")
        packets = packetizer.create_packets(
            processed_audio, target_sample_rate, frame_size_ms
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Packets", len(packets))
        with col2:
            samples_per_packet = int(target_sample_rate * frame_size_ms / 1000)
            st.metric("Samples per Packet", samples_per_packet)
        with col3:
            packet_rate = 1000 / frame_size_ms
            st.metric("Packet Rate", f"{packet_rate:.1f} packets/sec")

        progress_bar.progress(40)

        # Step 3: Network simulation
        status_text.text("🌐 **Step 3/5: Simulating network impairments...**")
        delivered_packets = network_sim.simulate(
            packets=packets,
            base_delay_ms=base_delay_ms,
            jitter_ms=jitter_ms,
            loss_model_config=loss_model_config,
            reorder_prob=reorder_prob,
            delay_distribution=delay_distribution,
        )

        # Calculate network statistics
        network_stats = network_sim.calculate_statistics(packets, delivered_packets)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Packets Sent", network_stats["sent"])
        with col2:
            st.metric("Packets Delivered", network_stats["delivered"])
        with col3:
            loss_pct = network_stats["loss_rate"] * 100
            st.metric("Network Loss", f"{loss_pct:.1f}%")
        with col4:
            st.metric("Avg Delay", f"{network_stats['avg_delay_ms']:.1f} ms")

        progress_bar.progress(60)

        # Step 4: Receiver processing
        status_text.text("🎚️ **Step 4/5: Processing at receiver...**")
        reconstructed, reception_stats = receiver.process(
            packets=delivered_packets,
            frame_size_ms=frame_size_ms,
            sample_rate=target_sample_rate,
            playout_delay_ms=playout_delay_ms,
            max_buffer_ms=max_buffer_ms,
            concealment=concealment,
            expected_packets=len(packets),
        )

        # Display receiver statistics
        st.subheader("🎚️ Receiver Statistics")
        rec_col1, rec_col2, rec_col3, rec_col4 = st.columns(4)
        with rec_col1:
            st.metric("On-time Packets", reception_stats["received_on_time"])
        with rec_col2:
            st.metric("Late Packets", reception_stats["received_late"])
        with rec_col3:
            st.metric("Lost Packets", reception_stats["lost"])
        with rec_col4:
            effective_loss = (
                reception_stats["lost"] + reception_stats["received_late"]
            ) / reception_stats["total_expected"]
            st.metric("Effective Loss", f"{effective_loss*100:.1f}%")

        progress_bar.progress(80)

        # Step 5: Quality analysis
        status_text.text("📊 **Step 5/5: Analyzing quality...**")

        # Ensure same length
        min_len = min(len(processed_audio), len(reconstructed))
        orig_trim = processed_audio[:min_len]
        recv_trim = reconstructed[:min_len]

        # Validate for PESQ
        pesq_warning = ""
        if target_sample_rate not in [8000, 16000]:
            pesq_warning = "PESQ only valid for 8kHz or 16kHz. Using resampled version."
        elif min_len / target_sample_rate < 1.0:
            pesq_warning = "Audio too short for reliable PESQ (<1 second)."

        if pesq_warning:
            st.warning(f"⚠️ {pesq_warning}")

        quality_metrics = metrics.calculate_all(
            orig_trim, recv_trim, target_sample_rate
        )
        interpretations = metrics.interpret_metrics(quality_metrics)

        progress_bar.progress(100)
        status_text.text("✅ **Simulation complete!**")

        # Display results
        disp_res(
            orig_trim,
            recv_trim,
            target_sample_rate,
            packets,
            delivered_packets,
            reception_stats,
            quality_metrics,
            interpretations,
            network_stats,
            visualizer,
            audio_processor,
        )

def disp_res(
    original_audio,
    received_audio,
    sample_rate,
    packets,
    delivered_packets,
    reception_stats,
    quality_metrics,
    interpretations,
    network_stats,
    visualizer,
    audio_processor,
):
    """Display simulation results."""

    # Audio comparison
    st.subheader("🎵 Audio Comparison")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original Audio**")
        original_bytes = audio_processor.save_audio_to_bytes(
            original_audio, sample_rate
        )
        st.audio(original_bytes.getvalue(), format="audio/wav")
        st.caption(
            f"Duration: {len(original_audio)/sample_rate:.2f}s | {len(original_audio):,} samples"
        )

    with col2:
        st.markdown("**Received Audio**")
        received_bytes = audio_processor.save_audio_to_bytes(
            received_audio, sample_rate
        )
        st.audio(received_bytes.getvalue(), format="audio/wav")
        st.caption(
            f"Duration: {len(received_audio)/sample_rate:.2f}s | {len(received_audio):,} samples"
        )

    # Quality metrics
    st.subheader("📈 Quality Metrics")

    metric_cols = st.columns(4)
    with metric_cols[0]:
        mse = quality_metrics["mse"]
        st.metric("MSE", f"{mse:.2e}", delta=interpretations["mse"])
        st.caption("Mean Squared Error (lower = better)")

    with metric_cols[1]:
        snr = quality_metrics["snr_db"]
        if snr == float("inf"):
            st.metric("SNR", "∞ dB", delta="Perfect")
        elif snr == float("-inf"):
            st.metric("SNR", "-∞ dB", delta="No signal")
        else:
            st.metric("SNR", f"{snr:.1f} dB", delta=interpretations["snr"])
        st.caption("Signal-to-Noise Ratio")

    with metric_cols[2]:
        pesq = quality_metrics["pesq"]
        if pesq is not None:
            st.metric("PESQ", f"{pesq:.2f}", delta=interpretations["pesq"])
            st.caption("Perceptual Quality (1.0-4.5)")
        else:
            st.metric("PESQ", "N/A", delta="Not available")
            st.caption("Install pesq or use 8k/16k audio")

    with metric_cols[3]:
        corr = quality_metrics.get("correlation", 0)
        st.metric("Correlation", f"{corr:.3f}", delta=interpretations["correlation"])
        st.caption("Waveform correlation (1.0 = perfect)")

    # Packet statistics
    st.subheader("📦 Packet Delivery Analysis")

    stats_cols = st.columns(4)
    with stats_cols[0]:
        total = reception_stats["total_expected"]
        st.metric("Total Expected", total)

    with stats_cols[1]:
        loss = reception_stats["loss_rate"]
        st.metric("Lost", f"{loss:.1%}", delta=f"{reception_stats['lost']} packets")
        st.caption("Never arrived")

    with stats_cols[2]:
        late = reception_stats["late_rate"]
        st.metric(
            "Late", f"{late:.1%}", delta=f"{reception_stats['received_late']} packets"
        )
        st.caption("Arrived after deadline")

    with stats_cols[3]:
        on_time = reception_stats["on_time_rate"]
        st.metric(
            "On-time",
            f"{on_time:.1%}",
            delta=f"{reception_stats['received_on_time']} packets",
        )
        st.caption("Played successfully")

    # Network vs Receiver comparison
    st.subheader("🌐 Network vs Receiver Comparison")
    comp_cols = st.columns(2)

    with comp_cols[0]:
        st.markdown("**Network-Level Loss**")
        network_loss = network_stats["loss_rate"] * 100
        st.metric("", f"{network_loss:.1f}%")
        st.caption("Packets dropped in network")

    with comp_cols[1]:
        st.markdown("**Application-Level Loss**")
        app_loss = (
            (reception_stats["lost"] + reception_stats["received_late"])
            / reception_stats["total_expected"]
            * 100
        )
        st.metric(
            "",
            f"{app_loss:.1f}%",
            delta=f"+{app_loss - network_loss:.1f}%" if app_loss > network_loss else "",
        )
        st.caption("Lost + Late (what user hears)")

    # Visualizations
    st.subheader("📊 Visual Analysis")

    viz_tabs = st.tabs(
        ["Waveforms", "Packet Timeline", "Delay Analysis", "Packet Stats"]
    )

    with viz_tabs[0]:
        st.markdown(
            """
    **Original vs Received Waveforms**
    
    **Y-axis:** Normalized amplitude (-1 to 1)
    **X-axis:** Time in seconds
    
    **What to look for:**
    - **Identical waveforms:** Perfect transmission (no network issues)
    - **Zero/flat sections:** Packet loss (silence or concealment inserted)
    - **Time shifts:** Jitter buffer adjustment or misalignment
    - **Amplitude changes:** Clipping, normalization differences, or attenuation
    - **Discontinuities:** Late/discarded packets causing gaps
    - **Pattern changes:** Different concealment methods applied
    
    **Common issues and interpretations:**
    - **Scale mismatch:** If amplitudes don't match, check normalization in audio processing
    - **Time misalignment:** Caused by jitter buffer, lost packets, or clock drift
    - **Clipping:** Values outside [-1, 1] range indicate improper normalization
    - **Blocky appearance:** Too many samples - waveform downsampled for visibility
    
    **Expected behavior for a clean sine wave (test audio):**
    - Smooth, regular oscillations
    - Constant amplitude (unless network effects)
    - No discontinuities or flat sections
    - Time-aligned with original (minor shifts possible from buffering)
    
    **Waveform display settings:**
    - Downsampled to 5000 points maximum for clear visualization
    - Colors: Blue = Original, Purple = Received
    - Thin lines (0.5-1.0px) to show detail
    - Fixed Y-axis scale for fair comparison
    - Zoom insets for detailed view of first 0.1 seconds
    """
        )

        try:
            # Main comparison plot with sine wave visibility
            st.markdown("### 📊 Waveform Comparison")
            fig = visualizer.plot_waveforms_comparison(
                original_audio, received_audio, sample_rate
            )
            st.pyplot(fig)

            # Show detailed sine wave analysis in expander
            with st.expander("🔬 Detailed Sine Wave Analysis", expanded=False):
                st.markdown(
                    """
                **Sine Wave Structure Analysis:**
                This view shows the audio at different zoom levels to reveal the sine wave structure.
                - **Top:** Full audio overview (downsampled if needed)
                - **Middle:** Medium zoom (50ms) - shows several cycles
                - **Bottom:** Close-up (10ms) - shows individual samples with markers
                
                **What to check for network effects:**
                - **Missing cycles:** Packet loss causing complete waveform gaps
                - **Distorted shape:** Concealment methods altering the sine wave
                - **Amplitude reduction:** Attenuation from network impairments
                - **Phase shifts:** Timing differences from buffering
                """
                )

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### 🎵 Original Audio - Sine Wave Structure")
                    fig2 = visualizer.plot_waveform_as_sine(
                        original_audio, sample_rate, "Original Audio"
                    )
                    st.pyplot(fig2)

                with col2:
                    st.markdown("#### 🔊 Received Audio - Sine Wave Structure")
                    fig3 = visualizer.plot_waveform_as_sine(
                        received_audio, sample_rate, "Received Audio"
                    )
                    st.pyplot(fig3)

                # Show difference analysis
                st.markdown("#### ⚖️ Waveform Difference Analysis")
                st.markdown(
                    """
                **What this shows:**
                - **Top:** Original waveform (what was sent)
                - **Middle:** Received waveform (after network simulation)  
                - **Bottom:** Difference (error) between them
                - **Red areas:** Where audio was lost or distorted
                - **Flat zero difference:** Perfect reconstruction
                - **Large differences:** Significant quality degradation
                
                **Error metrics displayed:**
                - MSE: Mean Squared Error (overall error magnitude)
                - RMSE: Root Mean Squared Error (average error)
                - Max Error: Largest single sample error
                """
                )
                fig4 = visualizer.plot_waveform_difference(
                    original_audio, received_audio, sample_rate
                )
                st.pyplot(fig4)

            # Audio statistics for debugging
            with st.expander("📈 Audio Statistics", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Original Audio Stats:**")
                    st.write(f"- Samples: {len(original_audio):,}")
                    st.write(f"- Duration: {len(original_audio)/sample_rate:.3f}s")
                    st.write(f"- Min amplitude: {np.min(original_audio):.6f}")
                    st.write(f"- Max amplitude: {np.max(original_audio):.6f}")
                    st.write(f"- Mean: {np.mean(original_audio):.6f}")
                    st.write(f"- RMS: {np.sqrt(np.mean(original_audio**2)):.6f}")

                    # Show first 20 samples
                    st.markdown("**First 20 samples:**")
                    st.write(original_audio[:20].round(6))

                with col2:
                    st.markdown("**Received Audio Stats:**")
                    st.write(f"- Samples: {len(received_audio):,}")
                    st.write(f"- Duration: {len(received_audio)/sample_rate:.3f}s")
                    st.write(f"- Min amplitude: {np.min(received_audio):.6f}")
                    st.write(f"- Max amplitude: {np.max(received_audio):.6f}")
                    st.write(f"- Mean: {np.mean(received_audio):.6f}")
                    st.write(f"- RMS: {np.sqrt(np.mean(received_audio**2)):.6f}")

                    # Show first 20 samples
                    st.markdown("**First 20 samples:**")
                    st.write(received_audio[:20].round(6))

                # Calculate and show correlation
                min_len = min(len(original_audio), len(received_audio))
                if min_len > 10:
                    correlation = np.corrcoef(
                        original_audio[:min_len], received_audio[:min_len]
                    )[0, 1]
                    st.metric("Waveform Correlation", f"{correlation:.6f}")

                # Quick diagnostic plot
                st.markdown("**Diagnostic: First 100 Samples**")
                fig_diag, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

                # Original
                ax1.plot(
                    original_audio[:100], "b-", linewidth=2, marker="o", markersize=4
                )
                ax1.set_title("Original (first 100 samples)")
                ax1.set_ylabel("Amplitude")
                ax1.grid(True, alpha=0.3)

                # Received
                ax2.plot(
                    received_audio[:100], "r-", linewidth=2, marker="s", markersize=4
                )
                ax2.set_title("Received (first 100 samples)")
                ax2.set_xlabel("Sample Index")
                ax2.set_ylabel("Amplitude")
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig_diag)

        except Exception as e:
            st.error(f"Could not generate waveform plots: {e}")

            # Show raw data for debugging
            st.markdown("**Debug Information:**")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Audio:**")
                st.write(f"Shape: {original_audio.shape}")
                st.write(f"Data type: {original_audio.dtype}")
                st.write(f"Min: {np.min(original_audio):.8f}")
                st.write(f"Max: {np.max(original_audio):.8f}")
                st.write(f"Mean: {np.mean(original_audio):.8f}")
                st.write(f"Std: {np.std(original_audio):.8f}")

                # Try simple plot
                try:
                    fig, ax = plt.subplots(figsize=(10, 3))
                    ax.plot(original_audio[:500], "b-", alpha=0.7, linewidth=0.5)
                    ax.set_title("Original Audio (first 500 samples)")
                    ax.set_xlabel("Sample")
                    ax.set_ylabel("Amplitude")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                except:
                    st.write("Could not create simple plot")

            with col2:
                st.markdown("**Received Audio:**")
                st.write(f"Shape: {received_audio.shape}")
                st.write(f"Data type: {received_audio.dtype}")
                st.write(f"Min: {np.min(received_audio):.8f}")
                st.write(f"Max: {np.max(received_audio):.8f}")
                st.write(f"Mean: {np.mean(received_audio):.8f}")
                st.write(f"Std: {np.std(received_audio):.8f}")

                # Try simple plot
                try:
                    fig, ax = plt.subplots(figsize=(10, 3))
                    ax.plot(received_audio[:500], "r-", alpha=0.7, linewidth=0.5)
                    ax.set_title("Received Audio (first 500 samples)")
                    ax.set_xlabel("Sample")
                    ax.set_ylabel("Amplitude")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                except:
                    st.write("Could not create simple plot")

    with viz_tabs[1]:
        st.markdown(
            """
        **Packet Transmission Timeline**
        
        **What this shows:**
        - **Send time (gray triangles):** When packet was transmitted from sender
        - **Arrival time (circles/squares):** When packet arrived at receiver
        - **Green dashed line:** Playout start time (when first packet played)
        - **Red dashed line:** Late packet deadline (absolute cutoff)
        - **Blue dotted lines:** Expected playout times for each packet
        
        **Colors and markers:**
        - **Green circles:** On-time packets (arrived before deadline)
        - **Orange squares:** Late packets (arrived after deadline, discarded)
        - **Gray triangles:** Send times (reference for delay calculation)
        - **Gray lines:** Network delay for each packet
        
        **Interpretation:**
        - **Vertical clustering:** Similar arrival times (low jitter)
        - **Horizontal spreading:** Variable delays (high jitter)
        - **Missing arrival markers:** Lost packets (no arrival point)
        - **Squares after red line:** Packets discarded as too late
        """
        )

        try:
            fig = visualizer.plot_packet_timeline(
                delivered_packets,
                frame_size_ms=20,
                playout_delay_ms=100,
                max_buffer_ms=200,
            )
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not generate timeline plot: {e}")

    with viz_tabs[2]:
        if delivered_packets and len(delivered_packets) > 0:
            st.markdown(
                """
            **Network Delay Distribution**
            
            **Left: Histogram of packet delays**
            - Shows frequency of different delay values
            - **Red dashed line:** Mean delay (average)
            - **Green dotted line:** Median delay (50th percentile)
            - **Bars:** Number of packets with specific delay
            
            **Right: Cumulative Distribution Function (CDF)**
            - Shows probability of delay ≤ X
            - Steep curve = consistent delays
            - Gentle slope = highly variable delays
            - **Gray lines:** Percentiles (50%, 75%, 90%, 95%, 99%)
            
            **Interpretation:**
            - **Narrow histogram:** Low jitter (consistent delays)
            - **Wide histogram:** High jitter (variable delays)
            - **Steep CDF:** Most packets have similar delays
            - **Gradual CDF:** Wide range of delay values
            - **Right-skewed:** More chance of moderate-high delays
            
            **Key metrics:**
            - **Mean:** Average delay across all packets
            - **Median:** Middle value (50% below, 50% above)
            - **Standard deviation:** Measure of jitter
            - **Percentiles:** Delay values at specific probabilities
            """
            )

            try:
                fig = visualizer.plot_delay_histogram(delivered_packets)
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not generate delay plot: {e}")
        else:
            st.info("No packets delivered to analyze delays.")

    with viz_tabs[3]:
        st.markdown(
            """
        **Packet Delivery Statistics**
        
        **Left: Percentage breakdown (Pie Chart)**
        Shows relative proportions of packet fates:
        - **Green:** On-time packets (successfully played)
        - **Orange:** Late packets (arrived but too late)
        - **Red:** Lost packets (never arrived)
        
        **Right: Absolute counts (Bar Chart)**
        Shows actual packet numbers:
        - **Gray:** Total packets expected
        - **Green:** On-time packets
        - **Orange:** Late packets  
        - **Red:** Lost packets
        
        **Effective Loss Rate** = (Lost + Late) / Total Expected
        This is what the user actually experiences - packets either missing or too late.
        
        **Key insights:**
        - Large orange slice: Buffer too small for network jitter
        - Large red slice: Network loss rate too high
        - Mostly green: Good network conditions or well-tuned buffer
        - Significant difference between network loss and effective loss: Buffer converting delay to loss
        """
        )

        try:
            fig = visualizer.plot_packet_statistics(reception_stats)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not generate statistics plot: {e}")

    # Educational insights
    with st.expander("🎓 Educational Insights & Analysis", expanded=True):
        provide_educational_insights(reception_stats, quality_metrics, network_stats)

    # Download section
    st.subheader("💾 Download Results")

    download_cols = st.columns(3)
    with download_cols[0]:
        st.download_button(
            label="📥 Download Received Audio",
            data=received_bytes.getvalue(),
            file_name="received_audio.wav",
            mime="audio/wav",
            help="Download the reconstructed audio file",
        )

    with download_cols[1]:
        # Export simulation report
        report = generate_report(reception_stats, quality_metrics, network_stats)
        st.download_button(
            label="📄 Download Simulation Report",
            data=report,
            file_name="simulation_report.txt",
            mime="text/plain",
            help="Download detailed simulation results",
        )

    with download_cols[2]:
        # Export configuration
        config_report = generate_config_report(
            sample_rate,
            frame_size_ms,
            base_delay_ms,
            jitter_ms, 
            loss_model,
        )
        st.download_button(
            label="⚙️ Download Configuration",
            data=config_report,
            file_name="simulation_config.txt",
            mime="text/plain",
            help="Download simulation configuration",
        )

    footer()


def generate_report(stats_data, quality_metrics, network_stats):
    """Generate a text report of simulation results."""
    report_lines = [
        "=" * 70,
        "RTP VOICE NETWORK SIMULATOR - SIMULATION REPORT",
        "=" * 70,
        "",
        "SIMULATION SUMMARY:",
        "-" * 40,
        f"Audio Samples Compared: {quality_metrics.get('samples_compared', 'N/A')}",
        f"Audio Duration: {quality_metrics.get('duration_seconds', 0):.2f} seconds",
        f"Sample Rate: {quality_metrics.get('sample_rate', 'N/A')} Hz",
        "",
        "PACKET DELIVERY STATISTICS:",
        "-" * 40,
        f"Total Packets Expected: {stats_data.get('total_expected', 0)}",
        f"Packets On Time: {stats_data.get('received_on_time', 0)} ({stats_data.get('on_time_rate', 0):.2%})",
        f"Packets Late (discarded): {stats_data.get('received_late', 0)} ({stats_data.get('late_rate', 0):.2%})",
        f"Packets Lost (network): {stats_data.get('lost', 0)} ({stats_data.get('loss_rate', 0):.2%})",
        f"Effective Loss (heard by user): {(stats_data.get('lost', 0) + stats_data.get('received_late', 0)) / max(1, stats_data.get('total_expected', 1)):.2%}",
        "",
        "NETWORK PERFORMANCE:",
        "-" * 40,
        f"Average Delay: {network_stats.get('avg_delay_ms', 0):.1f} ms",
        f"Maximum Delay: {network_stats.get('max_delay_ms', 0):.1f} ms",
        f"Average Jitter: {network_stats.get('avg_jitter_ms', 0):.1f} ms",
        f"Network Loss Rate: {network_stats.get('loss_rate', 0):.2%}",
        "",
        "AUDIO QUALITY METRICS:",
        "-" * 40,
        f"MSE (Mean Squared Error): {quality_metrics.get('mse', 0):.2e}",
        f"SNR (Signal-to-Noise Ratio): {quality_metrics.get('snr_db', 0):.1f} dB",
        f"PESQ (Perceptual Quality): {quality_metrics.get('pesq', 'Not available')}",
        f"Correlation Coefficient: {quality_metrics.get('correlation', 0):.3f}",
        f"Spectral Distortion: {quality_metrics.get('spectral_distortion', 0):.2f} dB",
        "",
        "QUALITY INTERPRETATION GUIDE:",
        "-" * 40,
        "MSE SCALE:",
        "  < 1e-6: Excellent (near perfect)",
        "  1e-6 to 1e-4: Very Good",
        "  1e-4 to 1e-2: Good",
        "  1e-2 to 0.1: Fair",
        "  > 0.1: Poor",
        "",
        "SNR SCALE (for speech):",
        "  > 30 dB: Excellent",
        "  20-30 dB: Good",
        "  10-20 dB: Fair",
        "  < 10 dB: Poor",
        "",
        "PESQ SCALE (1.0-4.5):",
        "  4.0-4.5: Excellent (transparent)",
        "  3.5-4.0: Good",
        "  3.0-3.5: Fair (noticeable but acceptable)",
        "  2.0-3.0: Poor",
        "  1.0-2.0: Bad",
        "",
        "PACKET LOSS IMPACT:",
        "  < 1%: Excellent VoIP quality",
        "  1-3%: Good, some artifacts",
        "  3-5%: Fair, noticeable gaps",
        "  5-10%: Poor, difficult conversation",
        "  > 10%: Very poor, often unusable",
        "",
        "LATENCY GUIDELINES (one-way):",
        "  < 150 ms: Good for conversation",
        "  150-300 ms: Acceptable with echo control",
        "  > 300 ms: Difficult for interactive talk",
        "",
        "=" * 70,
        "Generated by RTP Voice Network Simulator",
        "Educational Tool for Network Protocols Course",
        "=" * 70,
    ]

    return "\n".join(report_lines)


def generate_config_report(
    sample_rate, frame_size_ms, base_delay, jitter_ms, loss_config
):
    """Generate configuration report."""
    config_lines = [
        "=" * 60,
        "SIMULATION CONFIGURATION",
        "=" * 60,
        "",
        "AUDIO SETTINGS:",
        f"  Sample Rate: {sample_rate} Hz",
        f"  Frame Size: {frame_size_ms} ms",
        f"  Samples per Packet: {int(sample_rate * frame_size_ms / 1000)}",
        "",
        "NETWORK SETTINGS:",
        f"  Base Delay: {base_delay} ms",
        f"  Jitter: {jitter_ms} ms",
        "",
        "LOSS MODEL:",
    ]

    if loss_config.get("type") == "random":
        config_lines.append(f"  Type: Random (Bernoulli)")
        config_lines.append(
            f"  Loss Probability: {loss_config.get('loss_rate', 0):.3%}"
        )
    elif loss_config.get("type") == "gilbert_elliott":
        config_lines.append(f"  Type: Gilbert-Elliott (2-State Markov)")
        config_lines.append(f"  P(Good→Bad): {loss_config.get('p_gb', 0):.3f}")
        config_lines.append(f"  P(Bad→Good): {loss_config.get('p_bg', 0):.3f}")
        config_lines.append(
            f"  Loss in Good State: {loss_config.get('loss_good', 0):.4%}"
        )
        config_lines.append(
            f"  Loss in Bad State: {loss_config.get('loss_bad', 0):.3%}"
        )
    else:
        config_lines.append(f"  Type: None (perfect network)")

    config_lines.extend(
        [
            "",
            "=" * 60,
            "End of Configuration",
            "=" * 60,
        ]
    )

    return "\n".join(config_lines)


def provide_educational_insights(stats, quality_metrics, network_stats):
    """Provide educational analysis of results."""

    st.markdown("### 📝 Quality Analysis")

    insights = []

    # Packet loss analysis
    loss_rate = stats["loss_rate"]
    effective_loss = (stats["lost"] + stats["received_late"]) / stats["total_expected"]

    if effective_loss < 0.01:
        insights.append(
            (
                "✅ **Excellent Packet Delivery:**",
                f"Only {effective_loss:.1%} effective loss (lost + late). Typical for good VoIP quality (MOS 4.0+).",
            )
        )
    elif effective_loss < 0.03:
        insights.append(
            (
                "⚠️ **Good Packet Delivery:**",
                f"{effective_loss:.1%} effective loss. Some audible artifacts may be present but acceptable (MOS 3.5-4.0).",
            )
        )
    elif effective_loss < 0.05:
        insights.append(
            (
                "⚠️ **Moderate Packet Delivery:**",
                f"{effective_loss:.1%} effective loss. Noticeable quality degradation, may affect conversation (MOS 3.0-3.5).",
            )
        )
    elif effective_loss < 0.10:
        insights.append(
            (
                "❌ **Poor Packet Delivery:**",
                f"{effective_loss:.1%} effective loss. Significant impact on conversation quality (MOS 2.5-3.0).",
            )
        )
    else:
        insights.append(
            (
                "❌ **Very Poor Packet Delivery:**",
                f"{effective_loss:.1%} effective loss. Likely unusable for voice communication (MOS < 2.5).",
            )
        )

    # SNR analysis
    snr = quality_metrics["snr_db"]
    if snr > 40:
        insights.append(
            (
                "✅ **Excellent SNR:**",
                f"{snr:.1f} dB. Very clean reconstruction, noise barely perceptible.",
            )
        )
    elif snr > 30:
        insights.append(
            ("✅ **Good SNR:**", f"{snr:.1f} dB. Good quality, suitable for telephony.")
        )
    elif snr > 20:
        insights.append(
            (
                "⚠️ **Fair SNR:**",
                f"{snr:.1f} dB. Noticeable noise but speech intelligible.",
            )
        )
    elif snr > 10:
        insights.append(
            ("⚠️ **Poor SNR:**", f"{snr:.1f} dB. Significant noise, quality affected.")
        )
    else:
        insights.append(
            ("❌ **Very Poor SNR:**", f"{snr:.1f} dB. Noise dominates signal.")
        )

    # PESQ analysis
    pesq_val = quality_metrics["pesq"]
    if pesq_val:
        if pesq_val > 4.0:
            insights.append(
                (
                    "✅ **Excellent PESQ:**",
                    f"{pesq_val:.2f}. Near-transparent quality, indistinguishable from original.",
                )
            )
        elif pesq_val > 3.5:
            insights.append(
                (
                    "✅ **Good PESQ:**",
                    f"{pesq_val:.2f}. Good quality, slight degradation perceptible.",
                )
            )
        elif pesq_val > 3.0:
            insights.append(
                (
                    "⚠️ **Fair PESQ:**",
                    f"{pesq_val:.2f}. Acceptable but noticeable quality reduction.",
                )
            )
        elif pesq_val > 2.5:
            insights.append(
                (
                    "⚠️ **Poor PESQ:**",
                    f"{pesq_val:.2f}. Poor quality, annoying but speech understandable.",
                )
            )
        else:
            insights.append(
                (
                    "❌ **Bad PESQ:**",
                    f"{pesq_val:.2f}. Very poor quality, difficult to understand.",
                )
            )
    else:
        insights.append(
            (
                "ℹ️ **PESQ Unavailable:**",
                "PESQ requires 8kHz or 16kHz sample rate and minimum 1 second audio. Install python-pesq package if needed.",
            )
        )

    # Late packet analysis
    late_rate = stats["late_rate"]
    if late_rate > 0.05:
        insights.append(
            (
                "⚠️ **High Late Packet Rate:**",
                f"{late_rate:.1%} packets arrived too late. Consider increasing playout delay or max buffer tolerance.",
            )
        )
    elif late_rate > 0:
        insights.append(
            (
                "ℹ️ **Some Late Packets:**",
                f"{late_rate:.1%} packets arrived late. Receiver buffer working to handle network jitter.",
            )
        )

    # Network vs application loss
    network_loss = network_stats.get("loss_rate", 0)
    app_loss = effective_loss
    if app_loss > network_loss:
        insights.append(
            (
                "🔍 **Buffer Converts Delay to Loss:**",
                f"Network loss: {network_loss:.1%} → Application loss: {app_loss:.1%}. "
                f"Receiver buffer converted {app_loss - network_loss:.1%} delayed packets to loss.",
            )
        )
    elif app_loss < network_loss:
        insights.append(
            (
                "🔍 **Buffer Recovers Some Packets:**",
                f"Network loss: {network_loss:.1%} → Application loss: {app_loss:.1%}. "
                f"Receiver buffer saved {network_loss - app_loss:.1%} packets that arrived late but within tolerance.",
            )
        )

    # Display insights
    for icon, text in insights:
        st.markdown(f"{icon} {text}")

    st.markdown(
        """
    ### 💡 Recommendations for Improvement
    
    1. **To reduce late packets:**
       - Increase playout delay (trade-off: increases latency)
       - Increase max buffer tolerance (trade-off: more late playback)
       - Reduce network jitter if possible (QoS, better routing)
       - Use adaptive jitter buffers that adjust to network conditions
    
    2. **To improve audio quality:**
       - Use 'repeat' or 'interpolate' concealment instead of 'zero'
       - Reduce packet loss through network QoS (priority queuing)
       - Consider forward error correction (FEC) for loss-prone networks
       - Use better codecs (Opus instead of PCM for compression & robustness)
       - Implement packet loss concealment algorithms (waveform substitution)
    
    3. **General optimization strategies:**
       - Balance playout delay vs. packet loss trade-off based on use case
       - Match buffer size to measured network jitter (2-4× jitter)
       - Monitor network conditions and adjust dynamically
       - Consider using redundant transmission for critical packets
       - Implement echo cancellation for high-latency scenarios
    
    4. **Network infrastructure improvements:**
       - Implement QoS (Quality of Service) for VoIP traffic
       - Use traffic shaping to smooth bursty traffic
       - Optimize routing paths for minimum delay and jitter
       - Upgrade network equipment with better buffers and scheduling
    """
    )

    st.markdown(
        """
    ### 🎓 Key Concepts Demonstrated
    
    - **RTP Packetization:** How continuous audio is divided into discrete packets with sequence numbers and timestamps
    - **Network Impairments:** Real-world effects (delay, jitter, loss, reordering) and their statistical models
    - **Dejitter Buffer:** Time-based compensation for network variability, converting variable delay to constant delay
    - **Loss Concealment:** Techniques to mask missing audio data and maintain perceptual quality
    - **Quality Metrics:** Objective (MSE, SNR) vs. perceptual (PESQ) evaluation methods
    - **Quality-Latency Trade-off:** Fundamental compromise in real-time communication systems
    - **Network vs. Application Layer:** How physical/network layer impairments propagate to application layer experience
    - **Buffer Management:** Strategies for handling variable network conditions while maintaining acceptable quality
    
    **Takeaway:** VoIP quality depends on careful tuning of multiple parameters to balance latency, packet loss, and audio quality for specific network conditions.
    """
    )


def footer():
    """Add a footer to the Streamlit app."""

    st.markdown("---")

    # Footer columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🎓 Educational Tool")
        st.markdown(
            """
        - RTP Packetization
        - Network Impairments
        - Dejitter Buffering
        - Audio Quality Analysis
        - Quality-Latency Trade-offs
        """
        )

    with col2:
        st.markdown("### 📚 Resources")
        st.markdown(
            """
        - [RFC 3550: RTP Protocol](https://tools.ietf.org/html/rfc3550)
        - [RFC 3551: RTP Profiles](https://tools.ietf.org/html/rfc3551)
        - [ITU-T P.862: PESQ Standard](https://www.itu.int/rec/T-REC-P.862)
        - [VoIP Network Guide](https://www.voip-info.org/)
        """
        )

    with col3:
        st.markdown("### 📕 Details")
        st.markdown(
            """
        **Course:** QOS  
        
        **Tool:** RTP Voice Network Simulator  
        
        **Purpose:** Educational Purposes
        
        **Name:** Cyusa Adnan Junior 
        """
        )

    # Copyright and links
    st.markdown("---")

    footer_cols = st.columns([2, 1, 1])

    with footer_cols[0]:
        st.caption(
            "©2025 CAJ | RTP Voice Network Simulator"
        )

    with footer_cols[1]:
        if st.button("📖 Documentation"):
            show_documentation()

    with footer_cols[2]:
        if st.button("🔄 Reset App"):
            st.rerun()


def show_documentation():
    """Show documentation in an expander."""
    with st.expander("📖 Documentation & References", expanded=True):
        st.markdown(
            """
        ### 📋 How to Use This Simulator
        
        1. **Upload Audio:** Select a WAV audio file (3-10 seconds works best)
        2. **Configure Parameters:** Adjust network conditions in sidebar
        3. **Run Simulation:** Click the 'Run Simulation' button
        4. **Analyze Results:** Review audio comparison, metrics, and visualizations
        
        ### ⚠️ Important Limitations
        
        **PESQ Limitations:**
        - Only valid for 8kHz (narrowband) and 16kHz (wideband)
        - Requires minimum 1 second of audio (3+ seconds recommended)
        - Results for other rates are approximations or unavailable
        
        **Simulation Limitations:**
        - Simplified network models (real networks more complex)
        - Basic loss concealment (real systems use advanced algorithms)
        - No adaptive jitter buffers (fixed buffer size)
        - PCM codec only (no compression like Opus, G.711)
        - No echo cancellation modeled
        
        ### 🎯 Learning Objectives
        
        **RTP Protocol**
        - Packetization of voice streams into RTP packets
        - Sequence numbering for ordering and loss detection
        - Timestamps for synchronization and playout timing
        - Payload type identification for different codecs
        
        **Network Impairments**
        - Propagation delay effects on conversation
        - Jitter (variable delay) impact on buffers
        - Packet loss models (Random & Gilbert-Elliott)
        - Packet reordering and its effects
        
        **Receiver Processing**
        - Dejitter buffer operation and trade-offs
        - Playout delay calculations and tuning
        - Loss concealment techniques and quality impact
        - Buffer management strategies
        
        **Quality Assessment**
        - Objective metrics (MSE, SNR, correlation)
        - Perceptual evaluation (PESQ scoring)
        - Packet delivery statistics and analysis
        - Network vs. application layer performance
        
        ### 🔧 Technical Details
        
        **Simulation Parameters:**
        - **Frame Size:** Duration of audio in each RTP packet (10-60ms)
        - **Base Delay:** Constant network propagation delay (0-500ms)
        - **Jitter:** Random delay variation with different distributions
        - **Loss Models:** Random (independent) or Gilbert-Elliott (bursty)
        - **Playout Delay:** Receiver buffer delay before playback (0-500ms)
        - **Max Buffer:** Late packet threshold (0-1000ms)
        
        **Quality Metrics:**
        - **MSE:** Mean Squared Error (lower = better, 0 = perfect)
        - **SNR:** Signal-to-Noise Ratio in dB (higher = better)
        - **PESQ:** Perceptual Evaluation of Speech Quality (1.0-4.5 scale)
        - **Correlation:** Waveform similarity (1.0 = identical)
        
        ### 📖 References
        
        1. **RFC 3550:** RTP: A Transport Protocol for Real-Time Applications
        2. **RFC 3551:** RTP Profile for Audio and Video Conferences
        3. **ITU-T P.862:** Perceptual evaluation of speech quality (PESQ)
        4. Perkins, C. (2003). RTP: Audio and Video for the Internet
        5. Wang, Y., & Zhu, Q. (1998). Error control and concealment for video communication
        6. ITU-T G.114: One-way transmission time
        7. ITU-T G.113: Transmission impairments
        
        ### 🐛 Troubleshooting
        
        **Common Issues:**
        - **No audio output:** Check if audio file is in WAV format (not MP3 renamed to .wav)
        - **Import errors:** Ensure all required packages are installed and directories have __init__.py
        - **Memory issues:** Use shorter audio files (<30 seconds)
        - **PESQ errors:** Use 8kHz or 16kHz sample rate and sufficient audio length
        - **Waveform display issues:** Audio may be downsampled for visualization clarity
        
        **Installation:**
        ```bash
        pip install streamlit numpy soundfile scipy matplotlib pandas librosa pesq
        ```
        
        ### 🚀 Future Enhancements
        
        Planned features:
        - Multiple codec support (G.711, Opus, G.729)
        - Adaptive jitter buffers with dynamic adjustment
        - Real-time network monitoring and adaptation
        - MOS (Mean Opinion Score) estimation from PESQ
        - Comparative analysis of different scenarios
        - Wireless network models (802.11, 4G/5G specific impairments)
        - Echo cancellation modeling
        - Network topology visualization
        """
        )

if __name__ == "__main__":
    app()
    footer()