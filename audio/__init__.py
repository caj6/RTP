"""
Audio processing module.
Exports audio processing classes.
"""

from .processor import AudioProcessor
from .codec import PCMCodec

__all__ = ["AudioProcessor", "PCMCodec"]

__version__ = "1.0.0"
