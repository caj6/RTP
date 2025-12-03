"""
Packet loss models for network simulation.
"""

import numpy as np
from typing import Optional


class LossModel:
    """Base class for packet loss models."""

    def should_drop(self, packet_seq: int) -> bool:
        """
        Determine if a packet should be dropped.

        Args:
            packet_seq: Packet sequence number

        Returns:
            bool: True if packet should be dropped
        """
        raise NotImplementedError


class NoLossModel(LossModel):
    """No packet loss."""

    def should_drop(self, packet_seq: int) -> bool:
        return False


class RandomLossModel(LossModel):
    """Independent random packet loss."""

    def __init__(self, loss_probability: float):
        """
        Args:
            loss_probability: Probability of packet loss (0-1)
        """
        if not 0 <= loss_probability <= 1:
            raise ValueError("Loss probability must be between 0 and 1")
        self.loss_probability = loss_probability

    def should_drop(self, packet_seq: int) -> bool:
        return np.random.rand() < self.loss_probability


class GilbertElliottModel(LossModel):
    """
    Gilbert-Elliott two-state Markov model.
    Good state (G): No loss
    Bad state (B): High loss probability
    """

    def __init__(self, p_gb: float, p_bg: float, loss_in_bad: float = 0.7):
        """
        Args:
            p_gb: Probability of transition from Good to Bad
            p_bg: Probability of transition from Bad to Good
            loss_in_bad: Loss probability in Bad state
        """
        self.p_gb = p_gb
        self.p_bg = p_bg
        self.loss_in_bad = loss_in_bad
        self.state = "G"  # Start in Good state

    def should_drop(self, packet_seq: int) -> bool:
        # State transition
        if self.state == "G":
            if np.random.rand() < self.p_gb:
                self.state = "B"
        else:  # state == 'B'
            if np.random.rand() < self.p_bg:
                self.state = "G"

        # Loss decision
        if self.state == "G":
            return False
        else:
            return np.random.rand() < self.loss_in_bad


class BurstLossModel(LossModel):
    """Model with burst losses of specified average length."""

    def __init__(self, average_burst_length: int, overall_loss_rate: float):
        """
        Args:
            average_burst_length: Average length of loss bursts
            overall_loss_rate: Desired overall loss rate
        """
        self.avg_burst = average_burst_length
        self.target_loss = overall_loss_rate

        # Calculate transition probabilities
        # p = probability to stay in loss state
        # q = probability to stay in good state
        # Solve: avg_burst = 1/(1-p) and loss_rate = (1-q)/(2-p-q)

        p = 1 - (1 / average_burst_length)
        q = 1 - (overall_loss_rate * (2 - p) / (1 - overall_loss_rate))

        # Ensure probabilities are valid
        p = max(0, min(1, p))
        q = max(0, min(1, q))

        self.p = p  # P(loss|loss)
        self.q = q  # P(good|good)
        self.state = "good"

    def should_drop(self, packet_seq: int) -> bool:
        # State transition
        if self.state == "good":
            if np.random.rand() > self.q:
                self.state = "loss"
                return True
            else:
                return False
        else:  # state == 'loss'
            if np.random.rand() > self.p:
                self.state = "good"
                return False
            else:
                return True


class LossModelFactory:
    """Factory to create loss models from configuration."""

    @staticmethod
    def create(config: dict) -> LossModel:
        """
        Create loss model from configuration.

        Args:
            config: Dictionary with 'type' and parameters

        Returns:
            LossModel instance
        """
        if not config or config.get("type") in ["none", None]:
            return NoLossModel()

        model_type = config.get("type", "none")

        if model_type == "random":
            return RandomLossModel(config.get("loss_rate", 0.05))

        elif model_type == "gilbert_elliott":
            return GilbertElliottModel(
                p_gb=config.get("p_gb", 0.02),
                p_bg=config.get("p_bg", 0.3),
                loss_in_bad=config.get("loss_bad", 0.6),
            )

        elif model_type == "burst":
            return BurstLossModel(
                average_burst_length=config.get("burst_length", 3),
                overall_loss_rate=config.get("loss_rate", 0.1),
            )

        else:
            raise ValueError(f"Unknown loss model type: {model_type}")
