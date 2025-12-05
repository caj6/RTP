"""
Receiver with dejitter buffer and loss concealment.
"""

import numpy as np
from typing import List, Tuple
from rtp.packet import RTPPacket
from rtp.packetizer import Packetizer


class Receiver:
    """Receiver implementation with dejitter buffer."""

    def __init__(self):
        self.packetizer = Packetizer()
        self.last_good_frame = None

    def process(
        self,
        packets: List[RTPPacket],
        frame_size_ms: int,
        sample_rate: int,
        playout_delay_ms: int = 100,
        max_buffer_ms: int = 200,
        concealment: str = "zero",
        expected_packets: int = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Process received packets with dejitter buffer.

        Args:
            packets: List of received packets (in arrival order)
            frame_size_ms: Original frame duration
            sample_rate: Audio sample rate
            playout_delay_ms: Initial buffering delay
            max_buffer_ms: Maximum allowed delay
            concealment: Loss concealment method
            expected_packets: Total packets expected (for stats)

        Returns:
            Tuple of (reconstructed audio, statistics)
        """
        if not packets:
            empty_audio = np.array([], dtype=np.float32)
            stats = self._empty_statistics(expected_packets)
            return empty_audio, stats

        # Calculate frame size in samples
        frame_samples = int(sample_rate * frame_size_ms / 1000)

        # Map packets by sequence number
        packet_map = {pkt.sequence: pkt for pkt in packets}

        # Determine playout schedule
        first_arrival = min(pkt.arrival_time for pkt in packets)
        playout_start = first_arrival + (playout_delay_ms / 1000.0)

        # Determine sequence range
        if expected_packets is not None:
            max_seq = expected_packets - 1
        else:
            max_seq = max(pkt.sequence for pkt in packets)

        # Process each expected packet
        frames = []
        stats = {
            "total_expected": max_seq + 1,
            "received_on_time": 0,
            "received_late": 0,
            "lost": 0,
            "late_timestamps": [],
            "on_time_seq": [],
            "late_seq": [],
            "lost_seq": [],
        }

        for seq in range(max_seq + 1):
            playout_time = playout_start + (seq * frame_size_ms / 1000.0)

            packet = packet_map.get(seq)

            if packet is None:
                # Packet lost
                stats["lost"] += 1
                stats["lost_seq"].append(seq)
                frame = self._conceal_loss(frame_samples, concealment)
            else:
                # Check if packet is too late
                if packet.arrival_time > playout_time + (max_buffer_ms / 1000.0):
                    # Packet is late
                    stats["received_late"] += 1
                    stats["late_seq"].append(seq)
                    stats["late_timestamps"].append(packet.arrival_time)
                    frame = self._conceal_loss(frame_samples, concealment)
                else:
                    # Packet on time
                    stats["received_on_time"] += 1
                    stats["on_time_seq"].append(seq)
                    frame = self.packetizer.codec_util.pcm16_to_float(packet.payload)
                    self.last_good_frame = frame.copy()

            frames.append(frame)

        # Calculate rates
        total = stats["total_expected"]
        stats["on_time_rate"] = stats["received_on_time"] / total
        stats["late_rate"] = stats["received_late"] / total
        stats["loss_rate"] = stats["lost"] / total

        # Concatenate frames
        if frames:
            reconstructed = np.concatenate(frames)
        else:
            reconstructed = np.array([], dtype=np.float32)

        return reconstructed, stats

    def _conceal_loss(self, frame_size: int, method: str) -> np.ndarray:
        """
        Generate concealment frame for lost packet.

        Args:
            frame_size: Number of samples in frame
            method: Concealment method

        Returns:
            Concealment frame
        """
        if method == "repeat" and self.last_good_frame is not None:
            return self.last_good_frame.copy()
        elif method == "interpolate" and self.last_good_frame is not None:
            # Simple interpolation (repeat with fade)
            return self.last_good_frame.copy() * 0.5
        else:
            return np.zeros(frame_size, dtype=np.float32)

    def _empty_statistics(self, expected_packets: int = None) -> dict:
        """Return empty statistics when no packets received."""
        if expected_packets is None:
            expected_packets = 0

        return {
            "total_expected": expected_packets,
            "received_on_time": 0,
            "received_late": 0,
            "lost": expected_packets,
            "late_timestamps": [],
            "on_time_seq": [],
            "late_seq": [],
            "lost_seq": [],
            "on_time_rate": 0.0,
            "late_rate": 0.0,
            "loss_rate": 1.0 if expected_packets > 0 else 0.0,
        }
