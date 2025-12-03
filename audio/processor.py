"""
Audio loading and processing utilities.
"""

import io
import numpy as np
import soundfile as sf
from scipy.signal import resample
import warnings


class AudioProcessor:
    """Handles audio file loading and preprocessing."""

    def __init__(self):
        self.supported_formats = [".wav", ".flac"]


def load_audio(self, audio_file):
    """
    Load audio file from various formats using librosa.
    """
    try:
        import librosa

        # If it's a file upload object
        if hasattr(audio_file, "read"):
            audio_bytes = audio_file.read()

            # Save to temporary file
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            # Load with librosa
            audio_data, sample_rate = librosa.load(tmp_path, sr=None, mono=False)

            # Clean up temp file
            import os

            os.unlink(tmp_path)

        else:
            # Direct file path
            audio_data, sample_rate = librosa.load(audio_file, sr=None, mono=False)

        return audio_data, sample_rate

    except ImportError:
        # Fallback to soundfile for WAV only
        warnings.warn("librosa not installed. Only WAV files supported.")
        return self._load_wav_only(audio_file)

    def process_audio(self, audio_data, original_rate, target_rate):
        """
        Process audio: convert to mono, resample, normalize.

        Args:
            audio_data: Input audio array
            original_rate: Original sample rate
            target_rate: Target sample rate

        Returns:
            processed_audio: Mono, resampled float32 array
        """
        # Convert to mono if needed
        audio_data = self._to_mono(audio_data)

        # Convert to float32 if needed
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
            # Normalize if integer type
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / np.max(np.abs(audio_data))

        # Resample if needed
        if original_rate != target_rate:
            audio_data = self._resample(audio_data, original_rate, target_rate)

        # Normalize to prevent clipping
        max_val = np.max(np.abs(audio_data))
        if max_val > 0.95:  # Only normalize if close to clipping
            audio_data = audio_data / (max_val * 1.05)  # Add 5% headroom

        return audio_data

    def _to_mono(self, audio_data):
        """Convert stereo audio to mono."""
        if audio_data.ndim == 1:
            return audio_data
        elif audio_data.ndim == 2:
            # Average channels
            return np.mean(audio_data, axis=1)
        else:
            raise ValueError(f"Unsupported audio shape: {audio_data.shape}")

    def _resample(self, audio_data, orig_rate, target_rate):
        """Resample audio to target rate."""
        if orig_rate == target_rate:
            return audio_data

        duration = len(audio_data) / orig_rate
        new_length = int(duration * target_rate)

        return resample(audio_data, new_length)

    def save_audio_to_bytes(self, audio_data, sample_rate, format="WAV"):
        """
        Save audio data to bytes.

        Args:
            audio_data: Audio samples
            sample_rate: Sample rate
            format: Output format (only WAV supported)

        Returns:
            bytes: Audio file bytes
        """
        buffer = io.BytesIO()

        # Ensure audio is within range
        audio_data = np.clip(audio_data, -1.0, 1.0)

        # Convert to int16 for WAV format
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        sf.write(buffer, audio_data, sample_rate, format=format)
        buffer.seek(0)
        return buffer

    def get_audio_info(self, audio_data, sample_rate):
        """Get information about audio."""
        return {
            "duration": len(audio_data) / sample_rate,
            "samples": len(audio_data),
            "sample_rate": sample_rate,
            "channels": 1,  # Always mono after processing
            "bit_depth": 32,  # float32
            "max_amplitude": float(np.max(np.abs(audio_data))),
        }
