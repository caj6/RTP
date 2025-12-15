"""
Plot functions for the simulator.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List
from rtp.packet import RTPPacket

class Visualization:
    """Handles all plotting for the simulator."""

    def __init__(self, style="seaborn-v0_8-whitegrid"):
        """
        Initialize visualizer.

        Args:
            style: Matplotlib style to use
        """
        plt.style.use(style)
        self.colors = {
            "original": "#2E86AB",  # Blue
            "received": "#A23B72",  # Purple
            "on_time": "#4CAF50",  # Green
            "late": "#FF9800",  # Orange
            "lost": "#F44336",  # Red
            "background": "#F5F5F5",  # Light gray
            "grid": "#E0E0E0",  # Grid gray
        }

    def plot_waveforms_comparison(
        self, original: np.ndarray, received: np.ndarray, sample_rate: int
    ) -> plt.Figure:
        """
        Plot original vs received waveforms optimized for sine wave display.

        Args:
            original: Original audio samples
            received: Received audio samples
            sample_rate: Sample rate in Hz

        Returns:
            Matplotlib figure
        """
        if len(original) == 0 or len(received) == 0:
            return self._create_empty_plot("No audio data to display")

        # Ensure same length
        min_len = min(len(original), len(received))
        orig_trim = original[:min_len]
        recv_trim = received[:min_len]

        duration = min_len / sample_rate

        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), height_ratios=[2, 2, 3])

        # --- Plot 1: Full waveform overview ---
        time_full = np.arange(min_len) / sample_rate

        # Downsample for full view if too many points
        if len(time_full) > 10000:
            step = len(time_full) // 5000
            time_full = time_full[::step]
            orig_full = orig_trim[::step]
            recv_full = recv_trim[::step]
        else:
            orig_full = orig_trim
            recv_full = recv_trim

        axes[0].plot(
            time_full,
            orig_full,
            color=self.colors["original"],
            alpha=0.8,
            linewidth=0.8,
            label="Original",
        )
        axes[0].set_title("Original Audio - Full View", fontsize=14, fontweight="bold")
        axes[0].set_ylabel("Amplitude", fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=0, color="black", alpha=0.3, linewidth=0.5)
        axes[0].legend(loc="upper right")
        axes[0].set_xlim([0, duration])

        # --- Plot 2: Received full waveform ---
        axes[1].plot(
            time_full,
            recv_full,
            color=self.colors["received"],
            alpha=0.8,
            linewidth=0.8,
            label="Received",
        )
        axes[1].set_title("Received Audio - Full View", fontsize=14, fontweight="bold")
        axes[1].set_ylabel("Amplitude", fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color="black", alpha=0.3, linewidth=0.5)
        axes[1].legend(loc="upper right")
        axes[1].set_xlim([0, duration])

        # --- Plot 3: Zoomed comparison (first 0.05 seconds) ---
        zoom_duration = min(0.05, duration)  # Zoom to 50ms
        zoom_samples = int(zoom_duration * sample_rate)
        zoom_samples = min(zoom_samples, min_len)

        time_zoom = np.arange(zoom_samples) / sample_rate
        orig_zoom = orig_trim[:zoom_samples]
        recv_zoom = recv_trim[:zoom_samples]

        # Plot both on same axes for comparison
        axes[2].plot(
            time_zoom * 1000,
            orig_zoom,  # Convert to milliseconds
            color=self.colors["original"],
            alpha=0.9,
            linewidth=2,
            label="Original",
            marker="o" if zoom_samples < 100 else "",
            markersize=4 if zoom_samples < 100 else 0,
        )

        axes[2].plot(
            time_zoom * 1000,
            recv_zoom,
            color=self.colors["received"],
            alpha=0.7,
            linewidth=2,
            label="Received",
            linestyle="--",
            marker="s" if zoom_samples < 100 else "",
            markersize=4 if zoom_samples < 100 else 0,
        )

        axes[2].set_title(
            f"Zoomed Comparison - First {zoom_duration*1000:.0f}ms",
            fontsize=14,
            fontweight="bold",
        )
        axes[2].set_xlabel("Time (milliseconds)", fontsize=12)
        axes[2].set_ylabel("Amplitude", fontsize=12)
        axes[2].grid(True, alpha=0.3)
        axes[2].axhline(y=0, color="black", alpha=0.3, linewidth=0.5)
        axes[2].legend(loc="upper right")

        # Calculate and display sine wave properties
        def analyze_waveform(waveform, sr):
            """Analyze waveform for sine wave properties."""
            if len(waveform) < 10:
                return {"is_sine": False, "frequency": 0, "amplitude": 0}

            # Check if it looks like a sine wave
            # Calculate zero crossings
            zero_crossings = np.where(np.diff(np.sign(waveform)))[0]

            if len(zero_crossings) > 2:
                # Estimate frequency from zero crossings
                periods = len(zero_crossings) // 2
                if periods > 0:
                    total_time = len(waveform) / sr
                    frequency = periods / total_time

                    # Check amplitude consistency
                    peaks = np.abs(waveform)
                    amplitude = np.mean(peaks)
                    amplitude_std = np.std(peaks)

                    # Determine if it's a clean sine wave
                    is_sine = (amplitude_std / amplitude < 0.3) and (frequency > 20)

                    return {
                        "is_sine": is_sine,
                        "frequency": frequency,
                        "amplitude": amplitude,
                        "zero_crossings": len(zero_crossings),
                        "periods": periods,
                    }

            return {"is_sine": False, "frequency": 0, "amplitude": 0}

        # Analyze both waveforms
        orig_stats = analyze_waveform(orig_trim, sample_rate)
        recv_stats = analyze_waveform(recv_trim, sample_rate)

        # Add analysis text
        analysis_text = ""
        if orig_stats["is_sine"]:
            analysis_text += f"Original: Sine wave ~{orig_stats['frequency']:.0f}Hz, "
            analysis_text += f"Amp: {orig_stats['amplitude']:.3f}\n"
        else:
            analysis_text += "Original: Complex waveform\n"

        if recv_stats["is_sine"]:
            analysis_text += f"Received: Sine wave ~{recv_stats['frequency']:.0f}Hz, "
            analysis_text += f"Amp: {recv_stats['amplitude']:.3f}\n"
        else:
            analysis_text += "Received: Modified waveform\n"

        analysis_text += f"Duration: {duration:.3f}s, Sample rate: {sample_rate}Hz"

        # Add text box with analysis
        fig.text(
            0.5,
            0.02,
            analysis_text,
            fontsize=11,
            ha="center",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7),
        )

        plt.tight_layout(rect=[0, 0.05, 1, 0.98])
        return fig

    def plot_sine_wave_analysis(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        title: str = "Sine Wave Analysis",
    ) -> plt.Figure:
        """
        Detailed analysis plot for sine waves.

        Args:
            audio_data: Audio samples (should be a sine wave)
            sample_rate: Sample rate in Hz
            title: Plot title

        Returns:
            Matplotlib figure
        """
        if len(audio_data) == 0:
            return self._create_empty_plot("No audio data to display")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        time = np.arange(len(audio_data)) / sample_rate
        duration = len(audio_data) / sample_rate

        # Plot 1: Full waveform
        axes[0, 0].plot(time, audio_data, color="blue", alpha=0.7, linewidth=0.8)
        axes[0, 0].set_title(f"{title} - Full View", fontsize=12)
        axes[0, 0].set_xlabel("Time (s)")
        axes[0, 0].set_ylabel("Amplitude")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0, color="black", alpha=0.3, linewidth=0.5)

        # Plot 2: Zoomed view (first 0.02 seconds)
        zoom_samples = min(int(0.02 * sample_rate), len(audio_data))
        time_zoom = time[:zoom_samples]
        data_zoom = audio_data[:zoom_samples]

        axes[0, 1].plot(
            time_zoom * 1000,
            data_zoom,  # Convert to ms
            color="red",
            alpha=1.0,
            linewidth=2,
            marker="o",
            markersize=4,
        )
        axes[0, 1].set_title(f"{title} - Zoom (first 20ms)", fontsize=12)
        axes[0, 1].set_xlabel("Time (ms)")
        axes[0, 1].set_ylabel("Amplitude")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0, color="black", alpha=0.3, linewidth=0.5)

        # Plot 3: FFT analysis
        try:
            from scipy.fft import fft, fftfreq

            N = len(audio_data)
            yf = fft(audio_data)
            xf = fftfreq(N, 1 / sample_rate)[: N // 2]

            axes[1, 0].plot(
                xf,
                2.0 / N * np.abs(yf[0 : N // 2]),
                color="green",
                alpha=0.8,
                linewidth=1,
            )
            axes[1, 0].set_title("Frequency Spectrum", fontsize=12)
            axes[1, 0].set_xlabel("Frequency (Hz)")
            axes[1, 0].set_ylabel("Magnitude")
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_xlim([0, sample_rate / 2])

            # Mark dominant frequency
            dominant_idx = np.argmax(2.0 / N * np.abs(yf[0 : N // 2]))
            dominant_freq = xf[dominant_idx]
            dominant_mag = 2.0 / N * np.abs(yf[dominant_idx])

            axes[1, 0].plot(dominant_freq, dominant_mag, "ro", markersize=8)
            axes[1, 0].text(
                dominant_freq,
                dominant_mag,
                f"  {dominant_freq:.1f} Hz",
                verticalalignment="bottom",
            )

        except ImportError:
            axes[1, 0].text(
                0.5,
                0.5,
                "FFT not available",
                ha="center",
                va="center",
                transform=axes[1, 0].transAxes,
            )
            axes[1, 0].set_title("Frequency Spectrum (FFT not available)", fontsize=12)

        # Plot 4: Phase plot (current vs next sample)
        if len(audio_data) > 100:
            # Downsample for phase plot
            step = max(1, len(audio_data) // 1000)
            x = audio_data[::step]
            y = np.roll(x, -1)[:-1]

            axes[1, 1].scatter(x[:-1], y, color="purple", alpha=0.6, s=10)
            axes[1, 1].set_title("Phase Portrait", fontsize=12)
            axes[1, 1].set_xlabel("Sample[n]")
            axes[1, 1].set_ylabel("Sample[n+1]")
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].axhline(y=0, color="black", alpha=0.3, linewidth=0.5)
            axes[1, 1].axvline(x=0, color="black", alpha=0.3, linewidth=0.5)

            # Draw circle for ideal sine wave
            r = np.max(np.abs(audio_data))
            theta = np.linspace(0, 2 * np.pi, 100)
            axes[1, 1].plot(
                r * np.cos(theta), r * np.sin(theta), "r--", alpha=0.5, linewidth=1
            )
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "Not enough data for phase plot",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
            )

        # Calculate sine wave statistics
        if len(audio_data) > 10:
            try:
                from scipy.signal import find_peaks

                peaks, _ = find_peaks(
                    np.abs(audio_data),
                    height=np.max(np.abs(audio_data)) * 0.3,
                    distance=int(sample_rate / 1000),
                )  # At least 1ms between peaks

                if len(peaks) > 1:
                    avg_period = np.mean(np.diff(peaks)) / sample_rate
                    frequency = 1.0 / avg_period if avg_period > 0 else 0
                else:
                    frequency = 0
            except ImportError:
                frequency = 0

            amplitude = np.max(np.abs(audio_data))
            rms = np.sqrt(np.mean(audio_data**2))
            crest_factor = amplitude / rms if rms > 0 else 0

            # Calculate zero crossings for frequency estimation
            zero_crossings = np.where(np.diff(np.sign(audio_data)))[0]
            if len(zero_crossings) > 1:
                zero_freq = len(zero_crossings) / (2 * duration)
            else:
                zero_freq = 0

            stats_text = (
                f"Sine Wave Statistics:\n"
                f"• Estimated Frequency: {frequency:.1f} Hz\n"
                f"• Zero-crossing Frequency: {zero_freq:.1f} Hz\n"
                f"• Amplitude: {amplitude:.4f}\n"
                f"• RMS: {rms:.4f}\n"
                f"• Crest Factor: {crest_factor:.2f}\n"
                f"• Duration: {duration:.3f}s\n"
                f"• Samples: {len(audio_data):,}"
            )

            fig.text(
                0.98,
                0.02,
                stats_text,
                fontsize=9,
                verticalalignment="bottom",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        plt.tight_layout()
        return fig

    def plot_detailed_waveform(
        self, audio_data: np.ndarray, sample_rate: int, title: str = "Audio Waveform"
    ) -> plt.Figure:
        """
        Plot detailed waveform with zoom capabilities.

        Args:
            audio_data: Audio samples
            sample_rate: Sample rate in Hz
            title: Plot title

        Returns:
            Matplotlib figure
        """
        if len(audio_data) == 0:
            return self._create_empty_plot("No audio data to display")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

        # Downsample for full view if too many points
        time_full = np.arange(len(audio_data)) / sample_rate
        if len(time_full) > 10000:
            step = len(time_full) // 10000
            time_full = time_full[::step]
            audio_full = audio_data[::step]
        else:
            audio_full = audio_data

        # Full waveform with thinner line
        ax1.plot(
            time_full,
            audio_full,
            color=self.colors["original"],
            alpha=0.7,
            linewidth=0.5,  # Very thin line
        )

        # Only add fill for short audio
        if len(time_full) < 2000:
            ax1.fill_between(
                time_full, 0, audio_full, color=self.colors["original"], alpha=0.3
            )

        ax1.set_title(f"{title} - Full View", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Amplitude", fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Zoom in on first 0.1 seconds to show sine wave structure
        zoom_duration = min(0.1, len(audio_data) / sample_rate)
        zoom_samples = int(zoom_duration * sample_rate)
        zoom_samples = min(zoom_samples, len(audio_data))

        time_zoom = np.arange(zoom_samples) / sample_rate
        audio_zoom = audio_data[:zoom_samples]

        # Plot with markers for very short segments
        if zoom_samples < 100:
            # Show markers for individual points
            ax2.plot(
                time_zoom,
                audio_zoom,
                color=self.colors["received"],
                alpha=0.8,
                linewidth=1.5,
                marker="o",
                markersize=4,
                markerfacecolor=self.colors["received"],
                markeredgecolor="black",
                markeredgewidth=0.5,
            )
        else:
            # Regular line plot
            ax2.plot(
                time_zoom,
                audio_zoom,
                color=self.colors["received"],
                alpha=0.8,
                linewidth=1.0,
            )

        ax2.fill_between(
            time_zoom,
            0,
            audio_zoom,
            color=self.colors["received"],
            alpha=0.3,
        )
        ax2.set_title(
            f"{title} - First {zoom_duration:.3f}s Zoom", fontsize=14, fontweight="bold"
        )
        ax2.set_xlabel("Time (seconds)", fontsize=12)
        ax2.set_ylabel("Amplitude", fontsize=12)
        ax2.grid(True, alpha=0.3)

        # Add zero lines
        ax1.axhline(y=0, color="black", alpha=0.3, linewidth=0.5, linestyle="-")
        ax2.axhline(y=0, color="black", alpha=0.3, linewidth=0.5, linestyle="-")

        # Calculate detailed statistics
        rms = np.sqrt(np.mean(audio_data**2))
        peak = np.max(np.abs(audio_data))
        crest_factor = peak / rms if rms > 0 else 0

        # Calculate frequency content (simplified)
        if len(audio_data) > 100:
            from scipy.signal import welch

            frequencies, power = welch(
                audio_data, fs=sample_rate, nperseg=min(1024, len(audio_data))
            )
            dominant_freq = frequencies[np.argmax(power)]
        else:
            dominant_freq = 0

        stats_text = (
            f"Waveform Statistics:\n"
            f"• Duration: {len(audio_data)/sample_rate:.3f}s\n"
            f"• Samples: {len(audio_data):,}\n"
            f"• RMS: {rms:.4f}\n"
            f"• Peak: {peak:.4f}\n"
            f"• Crest factor: {crest_factor:.2f}\n"
            f"• Dominant freq: {dominant_freq:.1f} Hz"
        )

        fig.text(
            0.98,
            0.02,
            stats_text,
            fontsize=9,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        return fig

    def plot_waveform_as_sine(
        self, audio_data: np.ndarray, sample_rate: int, title: str = "Audio Waveform"
    ) -> plt.Figure:
        """
        Plot audio waveform emphasizing sine wave structure.

        Args:
            audio_data: Audio samples
            sample_rate: Sample rate in Hz
            title: Plot title

        Returns:
            Matplotlib figure
        """
        if len(audio_data) == 0:
            return self._create_empty_plot("No audio data to display")

        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        duration = len(audio_data) / sample_rate

        # 1. Full view with downsampling
        if len(audio_data) > 10000:
            step = len(audio_data) // 5000
            time_full = np.arange(0, len(audio_data), step) / sample_rate
            audio_full = audio_data[::step]
        else:
            time_full = np.arange(len(audio_data)) / sample_rate
            audio_full = audio_data

        axes[0].plot(
            time_full,
            audio_full,
            color=self.colors["original"],
            alpha=0.7,
            linewidth=0.8,
        )
        axes[0].set_title(f"{title} - Overview", fontsize=12, fontweight="bold")
        axes[0].set_ylabel("Amplitude")
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=0, color="black", alpha=0.3, linewidth=0.5)

        # 2. Medium zoom (first 0.05 seconds or 5 cycles of 100Hz)
        zoom1_duration = min(0.05, duration)
        zoom1_samples = int(zoom1_duration * sample_rate)
        zoom1_samples = min(zoom1_samples, len(audio_data))

        time_zoom1 = np.arange(zoom1_samples) / sample_rate
        audio_zoom1 = audio_data[:zoom1_samples]

        axes[1].plot(
            time_zoom1,
            audio_zoom1,
            color=self.colors["received"],
            alpha=0.9,
            linewidth=1.5,
            marker="o" if zoom1_samples < 50 else "",
            markersize=3 if zoom1_samples < 50 else 0,
        )
        axes[1].set_title(
            f"{title} - Medium Zoom ({zoom1_duration*1000:.0f}ms)",
            fontsize=12,
            fontweight="bold",
        )
        axes[1].set_ylabel("Amplitude")
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color="black", alpha=0.3, linewidth=0.5)

        # 3. Close-up zoom (first 0.01 seconds or 1 cycle of 100Hz)
        zoom2_duration = min(0.01, duration)
        zoom2_samples = int(zoom2_duration * sample_rate)
        zoom2_samples = min(zoom2_samples, len(audio_data))

        time_zoom2 = np.arange(zoom2_samples) / sample_rate
        audio_zoom2 = audio_data[:zoom2_samples]

        # Plot with markers for individual samples
        axes[2].plot(
            time_zoom2,
            audio_zoom2,
            color="#FF6B6B",
            alpha=1.0,
            linewidth=2.0,
            marker="o",
            markersize=5,
            markerfacecolor="#FF6B6B",
            markeredgecolor="black",
            markeredgewidth=0.5,
            label="Individual samples",
        )
        axes[2].set_title(
            f"{title} - Close-up ({zoom2_duration*1000:.0f}ms)",
            fontsize=12,
            fontweight="bold",
        )
        axes[2].set_xlabel("Time (seconds)", fontsize=12)
        axes[2].set_ylabel("Amplitude", fontsize=12)
        axes[2].grid(True, alpha=0.3)
        axes[2].axhline(y=0, color="black", alpha=0.3, linewidth=0.5)
        axes[2].legend(fontsize=10)

        # Add sample points as text for very short segments
        if zoom2_samples <= 20:
            for i, (t, val) in enumerate(zip(time_zoom2, audio_zoom2)):
                axes[2].text(
                    t,
                    val,
                    f"{i}",
                    fontsize=8,
                    ha="center",
                    va="bottom" if val >= 0 else "top",
                )

        # Calculate and display waveform properties
        from scipy.signal import find_peaks

        if len(audio_data) > 10:
            peaks, _ = find_peaks(
                np.abs(audio_data), height=np.max(np.abs(audio_data)) * 0.1
            )
            if len(peaks) > 1:
                period = np.mean(np.diff(peaks)) / sample_rate
                frequency = 1.0 / period if period > 0 else 0
            else:
                frequency = 0
        else:
            frequency = 0

        props_text = (
            f"Waveform Properties:\n"
            f"• Duration: {duration:.4f}s\n"
            f"• Sample rate: {sample_rate:,} Hz\n"
            f"• Total samples: {len(audio_data):,}\n"
            f"• Estimated frequency: {frequency:.1f} Hz\n"
            f"• Max amplitude: {np.max(np.abs(audio_data)):.4f}"
        )

        fig.text(
            0.98,
            0.02,
            props_text,
            fontsize=9,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
        )

        plt.tight_layout()
        return fig

    def plot_waveform_difference(
        self, original: np.ndarray, received: np.ndarray, sample_rate: int
    ) -> plt.Figure:
        """
        Plot difference between original and received waveforms.

        Args:
            original: Original audio samples
            received: Received audio samples
            sample_rate: Sample rate in Hz

        Returns:
            Matplotlib figure
        """
        if len(original) == 0 or len(received) == 0:
            return self._create_empty_plot("No audio data to display")

        # Ensure same length
        min_len = min(len(original), len(received))
        orig_trim = original[:min_len].copy()
        recv_trim = received[:min_len].copy()

        # Calculate difference (error)
        difference = orig_trim - recv_trim

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        time_axis = np.arange(min_len) / sample_rate

        # Downsample for visibility
        if len(time_axis) > 5000:
            step = len(time_axis) // 5000
            time_axis = time_axis[::step]
            orig_trim = orig_trim[::step]
            recv_trim = recv_trim[::step]
            difference = difference[::step]

        # Plot original
        ax1.plot(
            time_axis,
            orig_trim,
            color=self.colors["original"],
            alpha=0.7,
            linewidth=0.5,
        )
        ax1.set_ylabel("Original", fontsize=12)
        ax1.set_title("Waveform Comparison", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)

        # Plot received
        ax2.plot(
            time_axis,
            recv_trim,
            color=self.colors["received"],
            alpha=0.7,
            linewidth=0.5,
        )
        ax2.set_ylabel("Received", fontsize=12)
        ax2.grid(True, alpha=0.3)

        # Plot difference
        ax3.plot(time_axis, difference, color="red", alpha=0.7, linewidth=0.5)
        ax3.fill_between(time_axis, 0, difference, color="red", alpha=0.3)
        ax3.set_xlabel("Time (seconds)", fontsize=12)
        ax3.set_ylabel("Difference", fontsize=12)
        ax3.grid(True, alpha=0.3)

        # Calculate error metrics
        mse = np.mean(difference**2)
        rmse = np.sqrt(mse)
        max_error = np.max(np.abs(difference))

        error_text = (
            f"Error Metrics:\n"
            f"• MSE: {mse:.2e}\n"
            f"• RMSE: {rmse:.4f}\n"
            f"• Max error: {max_error:.4f}"
        )

        fig.text(
            0.98,
            0.02,
            error_text,
            fontsize=9,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.5),
        )

        plt.tight_layout()
        return fig

    def plot_packet_timeline(
        self,
        packets: List[RTPPacket],
        frame_size_ms: int,
        playout_delay_ms: int,
        max_buffer_ms: int,
    ) -> plt.Figure:
        """
        Plot packet transmission timeline.

        Args:
            packets: List of packets
            frame_size_ms: Frame duration
            playout_delay_ms: Playout delay
            max_buffer_ms: Maximum buffer time

        Returns:
            Matplotlib figure
        """
        if not packets:
            return self._create_empty_plot("No packets to display")

        fig, ax = plt.subplots(figsize=(12, 6))

        # Extract data
        seq_nums = [pkt.sequence for pkt in packets]
        send_times = [pkt.send_time * 1000 for pkt in packets]  # ms
        arrival_times = [pkt.arrival_time * 1000 for pkt in packets]  # ms

        # Calculate deadlines
        first_arrival = min(arrival_times)
        playout_start = first_arrival + playout_delay_ms
        deadline = playout_start + max_buffer_ms

        # Plot packet journeys
        for send, arrive, seq in zip(send_times, arrival_times, seq_nums):
            # Color based on arrival time
            if arrive <= playout_start + max_buffer_ms:
                color = self.colors["on_time"]
                marker = "o"
                size = 40
            else:
                color = self.colors["late"]
                marker = "s"
                size = 50

            # Plot send and arrival points
            ax.scatter(send, seq, color="gray", alpha=0.5, s=20, marker="^")
            ax.scatter(arrive, seq, color=color, alpha=0.7, s=size, marker=marker)

            # Draw connecting line
            ax.plot([send, arrive], [seq, seq], "gray", alpha=0.3, linewidth=1)

        # Draw reference lines
        ax.axvline(
            x=playout_start,
            color="green",
            linestyle="--",
            linewidth=2,
            label="Playout Start",
        )
        ax.axvline(
            x=deadline,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Late Deadline",
        )

        # Draw expected playout times
        for seq in seq_nums:
            expected_time = playout_start + (seq * frame_size_ms)
            ax.axvline(
                x=expected_time, color="blue", alpha=0.1, linestyle=":", linewidth=0.5
            )

        ax.set_xlabel("Time (ms)", fontsize=12)
        ax.set_ylabel("Packet Sequence Number", fontsize=12)
        ax.set_title("Packet Transmission Timeline", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_delay_histogram(self, packets: List[RTPPacket]) -> plt.Figure:
        """
        Plot histogram of packet delays.

        Args:
            packets: List of packets

        Returns:
            Matplotlib figure
        """
        if not packets:
            return self._create_empty_plot("No packets to display")

        # Calculate delays in ms
        delays = []
        for pkt in packets:
            if pkt.send_time is not None and pkt.arrival_time is not None:
                delay = (pkt.arrival_time - pkt.send_time) * 1000
                delays.append(delay)

        if not delays:
            return self._create_empty_plot("No delay data available")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Histogram
        ax1.hist(
            delays,
            bins=30,
            color=self.colors["original"],
            alpha=0.7,
            edgecolor="black",
        )
        ax1.set_xlabel("Delay (ms)", fontsize=12)
        ax1.set_ylabel("Number of Packets", fontsize=12)
        ax1.set_title("Packet Delay Distribution", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)

        # Add statistics
        mean_delay = np.mean(delays)
        std_delay = np.std(delays)
        median_delay = np.median(delays)

        ax1.axvline(
            mean_delay, color="red", linestyle="--", label=f"Mean: {mean_delay:.1f}ms"
        )
        ax1.axvline(
            median_delay,
            color="green",
            linestyle=":",
            label=f"Median: {median_delay:.1f}ms",
        )
        ax1.legend()

        # Cumulative distribution
        sorted_delays = np.sort(delays)
        cdf = np.arange(1, len(sorted_delays) + 1) / len(sorted_delays)

        ax2.plot(sorted_delays, cdf, color=self.colors["received"], linewidth=2)
        ax2.set_xlabel("Delay (ms)", fontsize=12)
        ax2.set_ylabel("Cumulative Probability", fontsize=12)
        ax2.set_title("Delay Cumulative Distribution", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        # Add percentile lines
        for percentile in [50, 75, 90, 95, 99]:
            delay_value = np.percentile(delays, percentile)
            ax2.axvline(delay_value, color="gray", linestyle="--", alpha=0.5)
            ax2.text(
                delay_value,
                0.05,
                f"{percentile}%",
                rotation=90,
                va="bottom",
                ha="right",
            )

        plt.tight_layout()
        return fig

    def plot_packet_statistics(self, stats: dict) -> plt.Figure:
        """
        Plot packet statistics pie chart and bar chart.

        Args:
            stats: Packet statistics dictionary

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Pie chart
        labels = ["On Time", "Late", "Lost"]
        sizes = [
            stats.get("on_time_rate", 0) * 100,
            stats.get("late_rate", 0) * 100,
            stats.get("loss_rate", 0) * 100,
        ]

        colors = [self.colors["on_time"], self.colors["late"], self.colors["lost"]]

        ax1.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            explode=(0.1, 0, 0),
        )
        ax1.set_title("Packet Delivery Status", fontsize=14, fontweight="bold")

        # Bar chart with counts
        categories = ["Total Expected", "On Time", "Late", "Lost"]
        counts = [
            stats.get("total_expected", 0),
            stats.get("received_on_time", 0),
            stats.get("received_late", 0),
            stats.get("lost", 0),
        ]

        bar_colors = [
            "gray",
            self.colors["on_time"],
            self.colors["late"],
            self.colors["lost"],
        ]

        bars = ax2.bar(categories, counts, color=bar_colors, alpha=0.8)
        ax2.set_title("Packet Counts", fontsize=14, fontweight="bold")
        ax2.set_ylabel("Number of Packets", fontsize=12)
        ax2.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(count)}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        return fig

    def _create_empty_plot(self, message: str) -> plt.Figure:
        """Create an empty plot with a message."""
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            fontsize=14,
            color="gray",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        return fig