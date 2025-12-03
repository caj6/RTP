"""
RTP packet handling module.
Exports packet classes and utilities.
"""

from .packet import RTPPacket
from .packetizer import Packetizer
from .loss_models import (
    LossModel,
    NoLossModel,
    RandomLossModel,
    GilbertElliottModel,
    BurstLossModel,
    LossModelFactory
)

__all__ = [
    'RTPPacket',
    'Packetizer',
    'LossModel',
    'NoLossModel',
    'RandomLossModel',
    'GilbertElliottModel',
    'BurstLossModel',
    'LossModelFactory'
]

__version__ = '1.0.0'