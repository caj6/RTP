"""
Quality metrics calculation.
"""

import numpy as np
import math
from typing import Dict, Optional

try:
    from pesq import pesq

    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False


class QualityMetrics:
    """Calculate audio quality metrics."""

    def calculate_all(
        self, original: np.ndarray, received: np.ndarray, sample_rate: int
    ) -> Dict:
        """
        Calculate all quality metrics.

        Args:
            original: Original audio samples
            received: Received audio samples
            sample_rate: Sample rate in Hz

        Returns:
            Dictionary of all metrics
        """
        # Ensure same length
        min_len = min(len(original), len(received))
        orig_trim = original[:min_len].copy()
        recv_trim = received[:min_len].copy()

        # Calculate basic metrics
        mse = self.calculate_mse(orig_trim, recv_trim)
        snr_db = self.calculate_snr(orig_trim, recv_trim)
        pesq_score = self.calculate_pesq(orig_trim, recv_trim, sample_rate)

        # Additional metrics
        correlation = self.calculate_correlation(orig_trim, recv_trim)
        spectral_distortion = self.calculate_spectral_distortion(
            orig_trim, recv_trim, sample_rate
        )

        return {
            "mse": mse,
            "snr_db": snr_db,
            "pesq": pesq_score,
            "correlation": correlation,
            "spectral_distortion": spectral_distortion,
            "samples_compared": min_len,
            "duration_seconds": min_len / sample_rate,
        }

    def calculate_mse(self, original: np.ndarray, received: np.ndarray) -> float:
        """Calculate Mean Squared Error."""
        if len(original) == 0 or len(received) == 0:
            return float("inf")

        return float(np.mean((original - received) ** 2))

    def calculate_snr(self, original: np.ndarray, received: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio in dB."""
        if len(original) == 0:
            return float("-inf")

        signal_power = np.mean(original**2)
        noise = original - received
        noise_power = np.mean(noise**2)

        if noise_power == 0:
            return float("inf")

        snr = 10 * math.log10(signal_power / noise_power)
        return snr

    def calculate_pesq(
        self, original: np.ndarray, received: np.ndarray, sample_rate: int
    ) -> Optional[float]:
        """Calculate PESQ score if available."""
        if not PESQ_AVAILABLE:
            return None

        if len(original) < 256:  # PESQ needs minimum length
            return None

        try:
            # PESQ expects specific sample rates
            if sample_rate == 8000:
                mode = "nb"  # narrowband
            elif sample_rate == 16000:
                mode = "wb"  # wideband
            else:
                # Resample to 16000 for PESQ
                from scipy.signal import resample_poly

                target_rate = 16000
                orig_resampled = resample_poly(original, target_rate, sample_rate)
                recv_resampled = resample_poly(received, target_rate, sample_rate)
                mode = "wb"
                original = orig_resampled
                received = recv_resampled
                sample_rate = target_rate

            pesq_score = pesq(sample_rate, original, received, mode)
            return float(pesq_score)

        except Exception:
            return None

    def calculate_correlation(
        self, original: np.ndarray, received: np.ndarray
    ) -> float:
        """Calculate correlation coefficient."""
        if len(original) < 2:
            return 0.0

        correlation = np.corrcoef(original, received)[0, 1]
        if np.isnan(correlation):
            return 0.0
        return float(correlation)

    def calculate_spectral_distortion(
        self,
        original: np.ndarray,
        received: np.ndarray,
        sample_rate: int,
        n_fft: int = 512,
    ) -> float:
        """Calculate average spectral distortion."""
        if len(original) < n_fft:
            return 0.0

        # Calculate spectrograms
        from scipy.signal import stft

        f_orig, t_orig, Zxx_orig = stft(
            original, fs=sample_rate, nperseg=n_fft, noverlap=n_fft // 2
        )
        f_recv, t_recv, Zxx_recv = stft(
            received, fs=sample_rate, nperseg=n_fft, noverlap=n_fft // 2
        )

        # Calculate magnitude spectrograms
        mag_orig = np.abs(Zxx_orig)
        mag_recv = np.abs(Zxx_recv)

        # Avoid division by zero
        mag_orig = np.maximum(mag_orig, 1e-10)
        mag_recv = np.maximum(mag_recv, 1e-10)

        # Calculate spectral distortion
        log_ratio = 20 * np.log10(mag_orig / mag_recv)
        distortion = np.sqrt(np.mean(log_ratio**2))

        return float(distortion)

    def interpret_metrics(self, metrics: Dict) -> Dict:
        """
        Provide interpretation of metrics.

        Args:
            metrics: Dictionary of calculated metrics

        Returns:
            Dictionary with interpretations
        """
        interpretations = {}

        # MSE interpretation
        mse = metrics.get("mse", float("inf"))
        if mse < 1e-6:
            interpretations["mse"] = "Excellent (near perfect)"
        elif mse < 1e-4:
            interpretations["mse"] = "Very Good"
        elif mse < 1e-2:
            interpretations["mse"] = "Good"
        elif mse < 0.1:
            interpretations["mse"] = "Fair"
        else:
            interpretations["mse"] = "Poor"

        # SNR interpretation
        snr = metrics.get("snr_db", float("-inf"))
        if snr > 40:
            interpretations["snr"] = "Excellent"
        elif snr > 30:
            interpretations["snr"] = "Very Good"
        elif snr > 20:
            interpretations["snr"] = "Good"
        elif snr > 10:
            interpretations["snr"] = "Fair"
        else:
            interpretations["snr"] = "Poor"

        # PESQ interpretation
        pesq_score = metrics.get("pesq")
        if pesq_score is not None:
            if pesq_score > 4.0:
                interpretations["pesq"] = "Excellent (transparent)"
            elif pesq_score > 3.5:
                interpretations["pesq"] = "Good"
            elif pesq_score > 3.0:
                interpretations["pesq"] = "Fair"
            elif pesq_score > 2.0:
                interpretations["pesq"] = "Poor"
            else:
                interpretations["pesq"] = "Bad"
        else:
            interpretations["pesq"] = "Not available"

        # Correlation interpretation
        corr = metrics.get("correlation", 0)
        if corr > 0.9:
            interpretations["correlation"] = "Excellent"
        elif corr > 0.7:
            interpretations["correlation"] = "Good"
        elif corr > 0.5:
            interpretations["correlation"] = "Moderate"
        else:
            interpretations["correlation"] = "Poor"

        return interpretations
