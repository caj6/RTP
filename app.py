"""
Main Streamlit application for RTP Voice Network Simulator.
"""

import streamlit as st
import io
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import modules
try:
    from audio.processor import AudioProcessor
    from rtp.packetizer import Packetizer
    from network.simulator import NetworkSimulator
    from network.receiver import Receiver
    from visualization.plots import Visualization
    from visualization.metrics import QualityMetrics

    imports_ok = True
except ImportError as e:
    st.error(f"Import error: {e}")
    st.info("Please make sure all module files are in the correct directories.")
    imports_ok = False


def main():
    """Main application entry point."""

    # Page configuration
    st.set_page_config(
        page_title="RTP Voice Network Simulator", page_icon="🎧", layout="wide"
    )

    # Title and description
    st.title("🎧 RTP Voice Network Simulator")
    st.markdown(
        """
    An educational tool to simulate RTP packet transmission over impaired networks.
    Upload audio, configure network conditions, and observe effects on voice quality.
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
            "Upload Audio File (WAV format recommended)",
            type=["wav"],  # Only WAV
            help="Upload a short WAV audio file (< 30 seconds). For MP3, convert to WAV first.",
        )

        # Processing settings
        st.subheader("⚙️ Processing Settings")
        target_sample_rate = st.selectbox(
            "Sample Rate (Hz)",
            [8000, 16000, 22050, 44100],
            index=1,
            help="Higher sample rates = better quality but larger packets",
        )
        frame_size_ms = st.slider(
            "Frame Size (ms)",
            10,
            60,
            20,
            5,
            help="Duration of audio in each RTP packet",
        )

        # Network settings
        st.subheader("🌐 Network Conditions")
        base_delay_ms = st.slider(
            "Base Delay (ms)", 0, 500, 50, 10, help="Constant network propagation delay"
        )
        jitter_ms = st.slider(
            "Jitter (ms)", 0, 200, 30, 5, help="Random delay variation"
        )

        # Loss model
        st.subheader("📉 Packet Loss")
        loss_model_type = st.selectbox(
            "Loss Model",
            ["None", "Random", "Gilbert-Elliott"],
            help="Choose packet loss model",
        )

        loss_model_config = {}
        if loss_model_type == "Random":
            loss_rate = st.slider(
                "Loss Rate", 0.0, 0.5, 0.05, 0.01, help="Probability of packet loss"
            )
            loss_model_config = {"type": "random", "loss_rate": loss_rate}
        elif loss_model_type == "Gilbert-Elliott":
            col1, col2 = st.columns(2)
            with col1:
                p_gb = st.slider("Good→Bad", 0.0, 0.2, 0.02, 0.01)
            with col2:
                p_bg = st.slider("Bad→Good", 0.0, 1.0, 0.3, 0.01)
            loss_bad = st.slider("Loss in Bad State", 0.0, 1.0, 0.6, 0.05)
            loss_model_config = {
                "type": "gilbert_elliott",
                "p_gb": p_gb,
                "p_bg": p_bg,
                "loss_bad": loss_bad,
            }
        else:
            loss_model_config = {"type": "none"}

        reorder_prob = st.slider(
            "Reordering Probability",
            0.0,
            0.3,
            0.02,
            0.01,
            help="Probability of packets arriving out of order",
        )

        # Receiver settings
        st.subheader("🎚️ Receiver Settings")
        playout_delay_ms = st.slider(
            "Playout Delay (ms)",
            0,
            500,
            100,
            10,
            help="Initial buffering delay before playback",
        )
        max_buffer_ms = st.slider(
            "Max Buffer (ms)",
            0,
            1000,
            200,
            10,
            help="Maximum allowed delay before dropping as late",
        )
        concealment = st.selectbox(
            "Loss Concealment",
            ["zero", "repeat"],
            help="Method for handling lost packets",
        )

        # Run button
        st.markdown("---")
        run_simulation = st.button(
            "🚀 Run Simulation", type="primary", use_container_width=True
        )

    # Main content area
    if audio_file is None:
        display_welcome_screen()
        return

    # Process audio file
    try:
        with st.spinner("Loading audio..."):
            # Load and process audio
            audio_data, sample_rate = audio_processor.load_audio(audio_file)
            audio_data = audio_processor.process_audio(
                audio_data, sample_rate, target_sample_rate
            )

            st.success(
                f"✅ Loaded audio: {len(audio_data)/target_sample_rate:.2f}s at {target_sample_rate}Hz"
            )

            # Display audio info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Duration", f"{len(audio_data)/target_sample_rate:.2f}s")
            with col2:
                st.metric("Samples", f"{len(audio_data):,}")
            with col3:
                st.metric("Sample Rate", f"{target_sample_rate} Hz")

            # Audio player for original
            st.subheader("🎵 Original Audio")
            original_bytes = audio_processor.save_audio_to_bytes(
                audio_data, target_sample_rate
            )
            st.audio(original_bytes.getvalue())

    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        st.info("Please try a different audio file (WAV format works best).")
        return

    # Run simulation when button is clicked
    if run_simulation:
        run_complete_simulation(
            audio_data=audio_data,
            sample_rate=target_sample_rate,
            frame_size_ms=frame_size_ms,
            base_delay_ms=base_delay_ms,
            jitter_ms=jitter_ms,
            loss_model_config=loss_model_config,
            reorder_prob=reorder_prob,
            playout_delay_ms=playout_delay_ms,
            max_buffer_ms=max_buffer_ms,
            concealment=concealment,
            audio_processor=audio_processor,
            packetizer=packetizer,
            network_sim=network_sim,
            receiver=receiver,
            visualizer=visualizer,
            metrics=metrics,
        )


def display_welcome_screen():
    """Display welcome screen when no audio is loaded."""
    st.info("👈 Please upload an audio file to begin simulation.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        ### 📋 How to Use:
        
        1. **Upload** an audio file (WAV or MP3)
        2. **Configure** network conditions in sidebar
        3. **Run** simulation to see results
        4. **Analyze** visualizations and metrics
        
        ### 🎯 Supported Formats:
        - WAV (recommended)
        - MP3
        - Mono or stereo (auto-converted to mono)
        - Any sample rate (auto-resampled)
        """
        )

    with col2:
        st.markdown(
            """
        ### 🎓 Educational Objectives:
        
        **Understand RTP Packetization:**
        - How voice is divided into packets
        - RTP header structure
        - Timestamp and sequence numbering
        
        **Study Network Impairments:**
        - Delay and jitter effects
        - Packet loss models
        - Packet reordering
        
        **Learn Receiver Processing:**
        - Dejitter buffer operation
        - Loss concealment techniques
        - Playback timing
        
        **Evaluate Voice Quality:**
        - Objective metrics (MSE, SNR)
        - Audio comparison
        """
        )

    # Quick example
    st.markdown("---")
    st.subheader("💡 Quick Example Parameters")
    example_cols = st.columns(4)
    with example_cols[0]:
        st.metric("Good VoIP", "Delay: 50ms")
    with example_cols[1]:
        st.metric("Moderate", "Jitter: 30ms")
    with example_cols[2]:
        st.metric("Poor", "Loss: 5%")
    with example_cols[3]:
        st.metric("Bad", "Loss: 20%")


def run_complete_simulation(
    audio_data,
    sample_rate,
    frame_size_ms,
    base_delay_ms,
    jitter_ms,
    loss_model_config,
    reorder_prob,
    playout_delay_ms,
    max_buffer_ms,
    concealment,
    audio_processor,
    packetizer,
    network_sim,
    receiver,
    visualizer,
    metrics,
):
    """Run the complete simulation pipeline."""

    # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Step 1: Packetization
    status_text.text("📦 Step 1/4: Packetizing audio...")
    packets = packetizer.create_packets(audio_data, sample_rate, frame_size_ms)
    progress_bar.progress(25)

    # Step 2: Network simulation
    status_text.text("🌐 Step 2/4: Simulating network...")
    delivered_packets = network_sim.simulate(
        packets=packets,
        base_delay_ms=base_delay_ms,
        jitter_ms=jitter_ms,
        loss_model_config=loss_model_config,
        reorder_prob=reorder_prob,
    )
    progress_bar.progress(50)

    # Step 3: Receiver processing
    status_text.text("🎚️ Step 3/4: Processing at receiver...")
    reconstructed, reception_stats = receiver.process(
        packets=delivered_packets,
        frame_size_ms=frame_size_ms,
        sample_rate=sample_rate,
        playout_delay_ms=playout_delay_ms,
        max_buffer_ms=max_buffer_ms,
        concealment=concealment,
        expected_packets=len(packets),
    )
    progress_bar.progress(75)

    # Step 4: Quality analysis
    status_text.text("📊 Step 4/4: Analyzing quality...")

    # Ensure same length
    min_len = min(len(audio_data), len(reconstructed))
    orig_trim = audio_data[:min_len]
    recv_trim = reconstructed[:min_len]

    quality_metrics = metrics.calculate_all(orig_trim, recv_trim, sample_rate)
    progress_bar.progress(100)
    status_text.text("✅ Simulation complete!")

    # Display results
    display_results(
        original_audio=orig_trim,
        received_audio=recv_trim,
        sample_rate=sample_rate,
        packets=packets,
        delivered_packets=delivered_packets,
        reception_stats=reception_stats,
        quality_metrics=quality_metrics,
        visualizer=visualizer,
        audio_processor=audio_processor,
    )


def display_results(
    original_audio,
    received_audio,
    sample_rate,
    packets,
    delivered_packets,
    reception_stats,
    quality_metrics,
    visualizer,
    audio_processor,
):
    """Display simulation results."""

    st.header("📊 Simulation Results")

    # Audio comparison section
    st.subheader("🎵 Audio Comparison")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original Audio**")
        original_bytes = audio_processor.save_audio_to_bytes(
            original_audio, sample_rate
        )
        st.audio(original_bytes.getvalue())
        st.caption(
            f"Duration: {len(original_audio)/sample_rate:.2f}s | Samples: {len(original_audio):,}"
        )

    with col2:
        st.markdown("**Received Audio**")
        received_bytes = audio_processor.save_audio_to_bytes(
            received_audio, sample_rate
        )
        st.audio(received_bytes.getvalue())
        st.caption(
            f"Duration: {len(received_audio)/sample_rate:.2f}s | Samples: {len(received_audio):,}"
        )

    # Quality metrics
    st.subheader("📈 Quality Metrics")

    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.metric("MSE", f"{quality_metrics['mse']:.2e}")
    with metric_cols[1]:
        snr_val = quality_metrics["snr_db"]
        st.metric("SNR", f"{snr_val:.1f} dB" if snr_val != float("inf") else "∞ dB")
    with metric_cols[2]:
        pesq_val = quality_metrics["pesq"]
        if pesq_val is not None:
            st.metric("PESQ", f"{pesq_val:.2f}")
        else:
            st.metric("PESQ", "N/A", help="Install pesq: pip install pesq")

    # Packet statistics
    st.subheader("📦 Packet Statistics")

    stats_cols = st.columns(4)
    with stats_cols[0]:
        st.metric("Total Packets", len(packets))
    with stats_cols[1]:
        loss_rate = reception_stats["loss_rate"]
        st.metric("Loss Rate", f"{loss_rate:.1%}")
    with stats_cols[2]:
        late_rate = reception_stats["late_rate"]
        st.metric("Late Rate", f"{late_rate:.1%}")
    with stats_cols[3]:
        on_time_rate = reception_stats["on_time_rate"]
        st.metric("On-time Rate", f"{on_time_rate:.1%}")

    # Visualizations
    st.subheader("📊 Visualizations")

    viz_tabs = st.tabs(["Waveforms", "Packet Stats", "Delay Analysis"])

    with viz_tabs[0]:
        fig_wave = visualizer.plot_waveforms_comparison(
            original_audio, received_audio, sample_rate
        )
        st.pyplot(fig_wave)

    with viz_tabs[1]:
        if delivered_packets:
            stats_fig = visualizer.plot_packet_statistics(reception_stats)
            st.pyplot(stats_fig)
        else:
            st.info("No packets delivered to display statistics.")

    with viz_tabs[2]:
        if delivered_packets and len(delivered_packets) > 0:
            try:
                fig_delay = visualizer.plot_delay_histogram(delivered_packets)
                st.pyplot(fig_delay)
            except Exception as e:
                st.warning(f"Could not generate delay plot: {e}")
        else:
            st.info("No packets delivered to analyze delays.")

    # Detailed statistics table
    with st.expander("📋 Detailed Statistics", expanded=False):
        stats_data = {
            "Total Packets Sent": len(packets),
            "Packets Delivered": len(delivered_packets),
            "Packets Lost": reception_stats["lost"],
            "Packets Late": reception_stats["received_late"],
            "Packets On Time": reception_stats["received_on_time"],
            "Loss Rate": f"{reception_stats['loss_rate']:.2%}",
            "Late Rate": f"{reception_stats['late_rate']:.2%}",
            "On-time Rate": f"{reception_stats['on_time_rate']:.2%}",
        }

        stats_df = pd.DataFrame(list(stats_data.items()), columns=["Metric", "Value"])
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    # Download section
    st.subheader("💾 Download Results")

    download_cols = st.columns(2)
    with download_cols[0]:
        st.download_button(
            label="📥 Download Received Audio",
            data=received_bytes.getvalue(),
            file_name="received_audio.wav",
            mime="audio/wav",
        )

    with download_cols[1]:
        # Export simulation report
        report = generate_report(stats_data, quality_metrics)
        st.download_button(
            label="📄 Download Simulation Report",
            data=report,
            file_name="simulation_report.txt",
            mime="text/plain",
        )

    # Educational insights
    with st.expander("🎓 Educational Insights & Analysis", expanded=True):
        provide_educational_insights(reception_stats, quality_metrics)


def generate_report(stats_data, quality_metrics):
    """Generate a text report of simulation results."""
    report_lines = [
        "=" * 50,
        "RTP Voice Network Simulator - Simulation Report",
        "=" * 50,
        "",
        "Packet Statistics:",
        "-" * 30,
    ]

    for key, value in stats_data.items():
        report_lines.append(f"{key}: {value}")

    report_lines.extend(
        [
            "",
            "Quality Metrics:",
            "-" * 30,
            f"MSE (Mean Squared Error): {quality_metrics['mse']:.2e}",
            f"SNR (Signal-to-Noise Ratio): {quality_metrics['snr_db']:.1f} dB",
            f"PESQ (Perceptual Quality): {quality_metrics['pesq'] or 'Not available'}",
            f"Correlation: {quality_metrics.get('correlation', 'N/A')}",
            "",
            "Interpretation:",
            "-" * 30,
            "MSE: Lower is better (0 is perfect)",
            "SNR: Higher is better (>20 dB is good)",
            "PESQ: 1.0-4.5 scale (higher is better)",
            "Packet Loss: <1% is acceptable for VoIP",
            "Late Packets: Treated as lost in real-time systems",
            "",
            "=" * 50,
        ]
    )

    return "\n".join(report_lines)


def provide_educational_insights(stats, metrics):
    """Provide educational analysis of results."""

    st.markdown("### 📝 Analysis")

    insights = []

    # Packet loss analysis
    loss_rate = stats["loss_rate"]
    if loss_rate < 0.01:
        insights.append(
            "✅ **Excellent**: Packet loss <1% - typical for good VoIP quality"
        )
    elif loss_rate < 0.05:
        insights.append("⚠️ **Moderate**: Packet loss 1-5% - some audible artifacts")
    else:
        insights.append(
            "❌ **Poor**: Packet loss >5% - significant quality degradation"
        )

    # SNR analysis
    snr = metrics["snr_db"]
    if snr > 30:
        insights.append("✅ **Excellent SNR**: >30 dB - very clean signal")
    elif snr > 20:
        insights.append("⚠️ **Good SNR**: 20-30 dB - acceptable for voice")
    else:
        insights.append("❌ **Poor SNR**: <20 dB - noticeable noise")

    # PESQ analysis
    pesq_val = metrics["pesq"]
    if pesq_val:
        if pesq_val > 4.0:
            insights.append("✅ **Excellent PESQ**: >4.0 - nearly transparent quality")
        elif pesq_val > 3.5:
            insights.append("⚠️ **Good PESQ**: 3.5-4.0 - good quality")
        elif pesq_val > 3.0:
            insights.append("⚠️ **Fair PESQ**: 3.0-3.5 - noticeable but acceptable")
        else:
            insights.append("❌ **Poor PESQ**: <3.0 - poor quality")

    # Late packet analysis
    late_rate = stats["late_rate"]
    if late_rate > 0.1:
        insights.append(
            f"⚠️ **High late packet rate**: {late_rate:.1%} - consider increasing playout delay"
        )

    # Display insights
    for insight in insights:
        st.write(insight)

    st.markdown(
        """
    ### 💡 Tips for Improvement:
    
    1. **Reduce Jitter**: Lower network jitter reduces late packets
    2. **Adjust Playout Delay**: Increase for more jitter tolerance (but increases latency)
    3. **Better Concealment**: 'repeat' often sounds better than 'zero'
    4. **Network QoS**: Implement Quality of Service for voice traffic
    5. **Frame Size**: Larger frames = less overhead but more sensitive to loss
    """
    )

    st.markdown(
        """
    ### 🎓 Key Concepts Demonstrated:
    
    - **RTP Packetization**: Audio divided into timestamped packets
    - **Network Impairments**: Real-world network conditions simulated
    - **Dejitter Buffer**: Compensates for variable network delays
    - **Loss Concealment**: Techniques to mask missing audio data
    - **Quality Metrics**: Objective measurement of audio quality
    """
    )


if __name__ == "__main__":
    main()
