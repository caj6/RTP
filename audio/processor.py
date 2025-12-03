"""
Audio loading and processing utilities.
"""

import io
import numpy as np
import soundfile as sf
from scipy.signal import resample
from pydub import AudioSegment
import warnings

class AudioProcessor:
    """Handles audio file loading and preprocessing."""
    
    def __init__(self):
        self.supported_formats = ['.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac']
    
    def load_audio(self, audio_file):
        """
        Load audio file from various formats.
        
        Args:
            audio_file: File object or path
            
        Returns:
            audio_data: numpy array (float32, -1 to 1)
            sample_rate: integer sample rate
        """
        try:
            # If it's a file upload object
            if hasattr(audio_file, 'read'):
                audio_bytes = audio_file.read()
                
                # Try soundfile first (better for wav/flac)
                try:
                    with io.BytesIO(audio_bytes) as buffer:
                        audio_data, sample_rate = sf.read(buffer)
                    return audio_data, sample_rate
                except:
                    # Fallback to pydub for other formats
                    return self._load_with_pydub(audio_bytes)
            else:
                # Assume it's a file path
                audio_data, sample_rate = sf.read(audio_file)
                return audio_data, sample_rate
                
        except Exception as e:
            raise ValueError(f"Failed to load audio file: {str(e)}")
    
    def _load_with_pydub(self, audio_bytes):
        """Load audio using pydub for unsupported formats."""
        try:
            # Create AudioSegment from bytes
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
            
            # Convert to numpy array
            samples = np.array(audio.get_array_of_samples())
            
            # Normalize to float32
            if audio.sample_width == 2:  # 16-bit
                samples = samples.astype(np.float32) / 32768.0
            elif audio.sample_width == 1:  # 8-bit
                samples = samples.astype(np.float32) / 128.0 - 1.0
            elif audio.sample_width == 3:  # 24-bit
                samples = samples.astype(np.float32) / 8388608.0
            elif audio.sample_width == 4:  # 32-bit
                samples = samples.astype(np.float32) / 2147483648.0
            else:
                samples = samples.astype(np.float32) / np.max(np.abs(samples))
            
            # Reshape if stereo
            if audio.channels == 2:
                samples = samples.reshape((-1, 2))
            
            return samples, audio.frame_rate
            
        except Exception as e:
            raise ValueError(f"Pydub failed to load audio: {str(e)}")
    
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
        if max_val > 1.0:
            audio_data = audio_data / max_val
        
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
    
    def save_audio_to_bytes(self, audio_data, sample_rate, format='WAV'):
        """
        Save audio data to bytes.
        
        Args:
            audio_data: Audio samples
            sample_rate: Sample rate
            format: Output format
            
        Returns:
            bytes: Audio file bytes
        """
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sample_rate, format=format)
        buffer.seek(0)
        return buffer
    
    def get_audio_info(self, audio_data, sample_rate):
        """Get information about audio."""
        return {
            'duration': len(audio_data) / sample_rate,
            'samples': len(audio_data),
            'sample_rate': sample_rate,
            'channels': 1 if audio_data.ndim == 1 else audio_data.shape[1],
            'bit_depth': 32,  # float32
            'max_amplitude': float(np.max(np.abs(audio_data)))
        }