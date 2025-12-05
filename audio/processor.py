"""
Audio loading and processing utilities.
"""

import io
import numpy as np
import soundfile as sf
from scipy.signal import resample

class AudioProcessor:
    """Handles audio file loading and preprocessing."""

    def __init__(self):
        self.supported_formats = [".wav"]

    def load_audio(self, audio_file):
        """
        Load audio file from WAV format.

        Args:
            audio_file: File object or path

        Returns:
            audio_data: numpy array (float32, -1 to 1)
            sample_rate: integer sample rate
        """
        try:
            # If it's a file upload object
            if hasattr(audio_file, "read"):
                audio_bytes = audio_file.read()

                # Use soundfile with BytesIO
                with io.BytesIO(audio_bytes) as buffer:
                    audio_data, sample_rate = sf.read(buffer)

                return audio_data, sample_rate
            else:
                # Assume it's a file path
                audio_data, sample_rate = sf.read(audio_file)
                return audio_data, sample_rate

        except Exception as e:
            raise ValueError(
                f"Failed to load WAV file: {str(e)}. Please use WAV format."
            )

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

        return audio_data

    def _to_mono(self, audio_data):
        """Convert stereo audio to mono."""
        if audio_data.ndim == 1:
            return audio_data
        elif audio_data.ndim == 2:
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

    def save_audio_to_bytes(self, audio_data, sample_rate):
        """
        Save audio data to bytes.

        Args:
            audio_data: Audio samples
            sample_rate: Sample rate

        Returns:
            bytes: Audio file bytes
        """
        buffer = io.BytesIO()

        # Ensure audio is within range
        audio_data = np.clip(audio_data, -1.0, 1.0)

        sf.write(buffer, audio_data, sample_rate, format="WAV")
        buffer.seek(0)
        return buffer
