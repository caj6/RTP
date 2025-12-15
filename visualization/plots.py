"""
Quality metrics calculation.
"""

import numpy as np
import math
from typing import Dict, Optional

# Disable PESQ due to import issues
PESQ_AVAILABLE = False


class QualityMetrics:
    """Calculate audio quality metrics."""

    def __init__(self):
        """Initialize quality metrics calculator."""
        pass

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

        # Calculate PESQ using fallback method
        pesq_score = self.calculate_pesq_fallback(orig_trim, recv_trim, sample_rate)

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
            "sample_rate": sample_rate,
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

    def calculate_pesq_fallback(
        self, original: np.ndarray, received: np.ndarray, sample_rate: int
    ) -> Optional[float]:
        """
        Fallback PESQ estimation when the PESQ library is not available.
        This provides reasonable approximations for educational purposes.
        """
        # Check for valid sample rate
        if sample_rate not in [8000, 16000]:
            return None

        # Check minimum duration
        min_len = min(len(original), len(received))
        duration = min_len / sample_rate
        if duration < 1.0:
            return None

        # Normalize audio
        orig_norm = self._normalize_for_pesq(original)
        recv_norm = self._normalize_for_pesq(received)

        try:
            # Calculate correlation
            correlation = np.corrcoef(orig_norm, recv_norm)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0

            # Calculate SNR
            signal_power = np.mean(orig_norm**2)
            noise_power = np.mean((orig_norm - recv_norm) ** 2)

            if noise_power > 0:
                snr_db = 10 * math.log10(signal_power / noise_power)
            else:
                snr_db = 50  # Perfect reconstruction

            # Estimate PESQ based on correlation and SNR
            # These formulas are approximations for educational purposes
            if correlation > 0.95:
                pesq_est = 4.0 + min(0.5, (correlation - 0.95) * 10)
            elif correlation > 0.85:
                pesq_est = 3.5 + (correlation - 0.85) * 5
            elif correlation > 0.70:
                pesq_est = 3.0 + (correlation - 0.70) * 3.33
            elif correlation > 0.50:
                pesq_est = 2.5 + (correlation - 0.50) * 2.5
            elif correlation > 0.30:
                pesq_est = 2.0 + (correlation - 0.30) * 2.5
            else:
                pesq_est = 1.0 + correlation * 3.33

            # Adjust based on SNR
            if snr_db > 40:
                pesq_est = min(4.5, pesq_est + 0.2)
            elif snr_db > 30:
                pesq_est = min(4.3, pesq_est + 0.1)
            elif snr_db < 10:
                pesq_est = max(1.0, pesq_est - 0.3)
            elif snr_db < 20:
                pesq_est = max(1.5, pesq_est - 0.2)

            # Clip to valid PESQ range
            pesq_est = max(1.0, min(4.5, pesq_est))

            return float(pesq_est)

        except Exception as e:
            print(f"Fallback PESQ calculation error: {e}")
            return None

    def _normalize_for_pesq(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio for PESQ calculation."""
        if len(audio) == 0:
            return audio

        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            target_rms = 0.05  # Approximate -26 dBov for speech
            audio = audio * (target_rms / rms)

        # Ensure within valid range
        audio = np.clip(audio, -1.0, 1.0)
        return audio

    def calculate_pesq(
        self, original: np.ndarray, received: np.ndarray, sample_rate: int
    ) -> Optional[float]:
        """Legacy method - use calculate_pesq_fallback instead."""
        return self.calculate_pesq_fallback(original, received, sample_rate)

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

        try:
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
        except Exception:
            return 0.0

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
        sample_rate = metrics.get("sample_rate", 0)

        if pesq_score is not None:
            if pesq_score > 4.0:
                interpretations["pesq"] = "Excellent (transparent)"
            elif pesq_score > 3.5:
                interpretations["pesq"] = "Good"
            elif pesq_score > 3.0:
                interpretations["pesq"] = "Fair"
            elif pesq_score > 2.5:
                interpretations["pesq"] = "Poor"
            elif pesq_score > 2.0:
                interpretations["pesq"] = "Bad"
            else:
                interpretations["pesq"] = "Very Bad"
        else:
            interpretations["pesq"] = "Not available (using fallback)"

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