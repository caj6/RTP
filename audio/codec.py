"""
Audio codec utilities for PCM encoding/decoding.
"""

import numpy as np

class PCMCodec:
    """PCM audio codec utilities."""
    
    @staticmethod
    def float_to_pcm16(samples):
        """
        Convert float32 samples (-1 to 1) to PCM16 bytes.
        
        Args:
            samples: numpy array of float32 samples
            
        Returns:
            bytes: PCM16 encoded audio
        """
        samples = np.clip(samples, -1.0, 1.0)
        int_samples = (samples * 32767).astype(np.int16)
        return int_samples.tobytes()
    
    @staticmethod
    def pcm16_to_float(pcm_bytes):
        """
        Convert PCM16 bytes to float32 samples.
        
        Args:
            pcm_bytes: bytes of PCM16 audio
            
        Returns:
            numpy array of float32 samples
        """
        int_samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        float_samples = int_samples.astype(np.float32) / 32768.0
        return float_samples
    
    @staticmethod
    def float_to_ulaw(samples):
        """
        Convert float32 to μ-law encoded bytes.
        Simplified version for educational purposes.
        """
        # This is a simplified μ-law implementation
        samples = np.clip(samples, -1.0, 1.0)
        # Scale and apply μ-law compression
        mu = 255
        scaled = np.sign(samples) * np.log1p(mu * np.abs(samples)) / np.log1p(mu)
        quantized = np.round((scaled + 1) * 127.5).astype(np.uint8)
        return quantized.tobytes()
    
    @staticmethod
    def ulaw_to_float(ulaw_bytes):
        """Convert μ-law bytes to float32."""
        quantized = np.frombuffer(ulaw_bytes, dtype=np.uint8)
        scaled = (quantized.astype(np.float32) / 127.5) - 1.0
        mu = 255
        samples = np.sign(scaled) * (1.0/mu) * ((1.0 + mu) ** np.abs(scaled) - 1.0)
        return samples