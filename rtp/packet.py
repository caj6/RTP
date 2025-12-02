"""
RTP packet implementation.
"""

import struct
import time

class RTPPacket:
    """RTP packet implementation for VoIP simulation."""
    
    # Payload type definitions
    PAYLOAD_TYPES = {
        'PCMU': 0,      # μ-law PCM
        'PCMA': 8,      # A-law PCM
        'L16': 11,      # Linear PCM 16-bit
        'G722': 9,      # G.722
        'G729': 18,     # G.729
    }
    
    def __init__(self, sequence, timestamp, payload, 
                 payload_type=11, marker=0, ssrc=None):
        """
        Initialize RTP packet.
        
        Args:
            sequence: Sequence number (16-bit)
            timestamp: RTP timestamp (32-bit)
            payload: Audio payload bytes
            payload_type: RTP payload type
            marker: Marker bit (for frame boundaries)
            ssrc: Synchronization source identifier
        """
        self.version = 2          # RTP version
        self.padding = 0          # Padding flag
        self.extension = 0        # Extension flag
        self.csrc_count = 0       # CSRC count
        self.marker = marker      # Marker bit
        self.payload_type = payload_type
        self.sequence = sequence
        self.timestamp = timestamp
        
        if ssrc is None:
            self.ssrc = 0x12345678  # Default SSRC
        else:
            self.ssrc = ssrc
            
        self.payload = payload
        self.send_time = None     # Simulation send time
        self.arrival_time = None  # Simulation arrival time
    
    def build_header(self):
        """
        Build RTP header bytes.
        
        Returns:
            bytes: 12-byte RTP header
        """
        # First byte: V, P, X, CC
        first_byte = (self.version << 6) | (self.padding << 5) | \
                    (self.extension << 4) | self.csrc_count
        
        # Second byte: M, PT
        second_byte = (self.marker << 7) | (self.payload_type & 0x7F)
        
        # Pack header
        header = struct.pack('!BBHII',
                           first_byte,
                           second_byte,
                           self.sequence,
                           self.timestamp,
                           self.ssrc)
        
        return header
    
    def to_bytes(self):
        """
        Convert entire packet to bytes.
        
        Returns:
            bytes: Complete RTP packet
        """
        return self.build_header() + self.payload
    
    @classmethod
    def from_bytes(cls, packet_bytes):
        """
        Parse RTP packet from bytes.
        
        Args:
            packet_bytes: Complete RTP packet bytes
            
        Returns:
            RTPPacket: Parsed packet object
        """
        if len(packet_bytes) < 12:
            raise ValueError("Packet too short for RTP header")
        
        # Parse header
        first_byte, second_byte, sequence, timestamp, ssrc = \
            struct.unpack('!BBHII', packet_bytes[:12])
        
        # Extract fields
        version = (first_byte >> 6) & 0x03
        padding = (first_byte >> 5) & 0x01
        extension = (first_byte >> 4) & 0x01
        csrc_count = first_byte & 0x0F
        marker = (second_byte >> 7) & 0x01
        payload_type = second_byte & 0x7F
        
        # Get payload
        header_size = 12 + (csrc_count * 4)
        if extension:
            # Skip extension header
            ext_len = struct.unpack('!H', packet_bytes[header_size+2:header_size+4])[0]
            header_size += 4 + (ext_len * 4)
        
        payload = packet_bytes[header_size:]
        
        # Create packet
        packet = cls(sequence, timestamp, payload, payload_type, marker, ssrc)
        packet.version = version
        packet.padding = padding
        packet.extension = extension
        packet.csrc_count = csrc_count
        
        return packet
    
    def __str__(self):
        """String representation of packet."""
        return (f"RTPPacket(seq={self.sequence}, ts={self.timestamp}, "
                f"pt={self.payload_type}, len={len(self.payload)})")
    
    def __repr__(self):
        return self.__str__()