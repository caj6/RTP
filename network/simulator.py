"""
Network simulation with impairments.
"""

import numpy as np
from typing import List
from rtp.packet import RTPPacket
from rtp.loss_models import LossModelFactory


class NetworkSimulator:
    """Simulates network impairments on RTP packets."""

    def __init__(self):
        self.loss_model = None

    def simulate(
        self,
        packets: List[RTPPacket],
        base_delay_ms: float = 50,
        jitter_ms: float = 30,
        loss_model_config: dict = None,
        reorder_prob: float = 0.0,
    ) -> List[RTPPacket]:
        """
        Apply network impairments to packets.

        Args:
            packets: List of RTP packets to transmit
            base_delay_ms: Constant propagation delay
            jitter_ms: Random delay variation (std dev)
            loss_model_config: Configuration for loss model
            reorder_prob: Probability of packet reordering

        Returns:
            List of packets that arrived (in arrival order)
        """
        # Handle None config
        if loss_model_config is None:
            loss_model_config = {}

        # Create loss model
        self.loss_model = LossModelFactory.create(loss_model_config)

        delivered_packets = []
        current_time = 0.0

        # Apply impairments to each packet
        for packet in packets:
            packet.send_time = current_time

            # Check for loss
            if self.loss_model.should_drop(packet.sequence):
                continue  # Packet lost

            # Calculate delay with jitter
            jitter = max(0, np.random.normal(0, jitter_ms / 1000.0))
            total_delay = base_delay_ms / 1000.0 + jitter
            packet.arrival_time = packet.send_time + total_delay

            delivered_packets.append(packet)

            # Packets sent at regular intervals (simulating 20ms frames)
            current_time += 0.020

        # Apply reordering
        if reorder_prob > 0 and delivered_packets:
            delivered_packets = self._apply_reordering(delivered_packets, reorder_prob)

        # Sort by arrival time
        delivered_packets.sort(key=lambda p: p.arrival_time)

        return delivered_packets

    def _apply_reordering(
        self, packets: List[RTPPacket], probability: float
    ) -> List[RTPPacket]:
        """
        Simulate packet reordering by swapping adjacent packets.

        Args:
            packets: List of packets
            probability: Probability of swapping adjacent packets

        Returns:
            List of packets with reordering applied
        """
        if len(packets) < 2:
            return packets

        result = packets.copy()

        for i in range(len(result) - 1):
            if np.random.rand() < probability:
                # Swap arrival times
                result[i].arrival_time, result[i + 1].arrival_time = (
                    result[i + 1].arrival_time,
                    result[i].arrival_time,
                )

        return result

    def calculate_statistics(
        self, original_packets: List[RTPPacket], delivered_packets: List[RTPPacket]
    ) -> dict:
        """
        Calculate network statistics.

        Args:
            original_packets: All packets sent
            delivered_packets: Packets that arrived

        Returns:
            Dictionary with statistics
        """
        total_sent = len(original_packets)
        total_delivered = len(delivered_packets)

        if total_delivered == 0:
            return {
                "sent": total_sent,
                "delivered": 0,
                "lost": total_sent,
                "loss_rate": 1.0,
                "avg_delay": 0,
                "max_delay": 0,
                "jitter": 0,
            }

        # Calculate delays
        delays = []
        for pkt in delivered_packets:
            if pkt.send_time is not None and pkt.arrival_time is not None:
                delay = (pkt.arrival_time - pkt.send_time) * 1000  # ms
                delays.append(delay)

        if delays:
            avg_delay = np.mean(delays)
            max_delay = np.max(delays)

            # Calculate jitter (RFC 3550)
            jitters = []
            for i in range(1, len(delays)):
                jitter = abs(delays[i] - delays[i - 1])
                jitters.append(jitter)

            avg_jitter = np.mean(jitters) if jitters else 0
        else:
            avg_delay = 0
            max_delay = 0
            avg_jitter = 0

        return {
            "sent": total_sent,
            "delivered": total_delivered,
            "lost": total_sent - total_delivered,
            "loss_rate": (total_sent - total_delivered) / total_sent,
            "avg_delay_ms": avg_delay,
            "max_delay_ms": max_delay,
            "avg_jitter_ms": avg_jitter,
            "delays_ms": delays,
        }
