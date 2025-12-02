"""
RTP packet creation and packetization.
"""

import numpy as np
from .packet import RTPPacket
from audio.codec import PCMCodec

class Packetizer:
    """Creates RTP packets from audio samples."""
    
    def __init__(self, codec='PCM16'):
        """
        Initialize packetizer.
        
        Args:
            codec: Audio codec ('PCM16', 'PCMU', 'PCMA')
        """
        self.codec = codec
        self.codec_util = PCMCodec()
        
    def create_packets(self, audio_samples, sample_rate, frame_duration_ms=20):
        """
        Convert audio samples to RTP packets.
        
        Args:
            audio_samples: Float32 audio samples
            sample_rate: Sample rate in Hz
            frame_duration_ms: Packet duration in milliseconds
            
        Returns:
            list: List of RTPPacket objects
        """
        # Calculate samples per frame
        samples_per_frame = int(sample_rate * frame_duration_ms / 1000)
        
        packets = []
        sequence_number = 0
        timestamp = 0
        
        # Split audio into frames
        for start_idx in range(0, len(audio_samples), samples_per_frame):
            frame = audio_samples[start_idx:start_idx + samples_per_frame]
            
            # Pad last frame if necessary
            if len(frame) < samples_per_frame:
                padding = samples_per_frame - len(frame)
                frame = np.pad(frame, (0, padding), mode='constant')
            
            # Encode frame based on codec
            if self.codec == 'PCM16':
                payload = self.codec_util.float_to_pcm16(frame)
                payload_type = RTPPacket.PAYLOAD_TYPES['L16']
            elif self.codec == 'PCMU':
                payload = self.codec_util.float_to_ulaw(frame)
                payload_type = RTPPacket.PAYLOAD_TYPES['PCMU']
            elif self.codec == 'PCMA':
                # A-law would go here
                payload = self.codec_util.float_to_pcm16(frame)
                payload_type = RTPPacket.PAYLOAD_TYPES['PCMA']
            else:
                raise ValueError(f"Unsupported codec: {self.codec}")
            
            # Create RTP packet
            # Set marker on first packet of talk spurt (simplified: every 10 packets)
            marker = 1 if sequence_number % 10 == 0 else 0
            
            packet = RTPPacket(
                sequence=sequence_number,
                timestamp=timestamp,
                payload=payload,
                payload_type=payload_type,
                marker=marker
            )
            
            packets.append(packet)
            
            # Update counters
            sequence_number += 1
            timestamp += samples_per_frame
        
        return packets
    
    def decode_packets(self, packets, codec='PCM16'):
        """
        Decode RTP packets back to audio samples.
        
        Args:
            packets: List of RTPPacket objects
            codec: Audio codec
            
        Returns:
            numpy array: Reconstructed audio samples
        """
        frames = []
        
        for packet in packets:
            if codec == 'PCM16':
                frame = self.codec_util.pcm16_to_float(packet.payload)
            elif codec == 'PCMU':
                frame = self.codec_util.ulaw_to_float(packet.payload)
            elif codec == 'PCMA':
                # A-law decoding would go here
                frame = self.codec_util.pcm16_to_float(packet.payload)
            else:
                raise ValueError(f"Unsupported codec: {codec}")
            
            frames.append(frame)
        
        if frames:
            return np.concatenate(frames)
        else:
            return np.array([], dtype=np.float32)