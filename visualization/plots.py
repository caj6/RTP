"""
Visualization functions for the simulator.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List
from rtp.packet import RTPPacket

class Visualization:
    """Handles all plotting for the simulator."""
    
    def __init__(self, style='ggplot'):
        """
        Initialize visualizer.
        
        Args:
            style: Matplotlib style to use
        """
        plt.style.use(style)
        self.colors = {
            'original': '#2E86AB',
            'received': '#A23B72',
            'on_time': '#4CAF50',
            'late': '#FF9800',
            'lost': '#F44336',
            'background': '#F5F5F5'
        }
    
    def plot_waveforms_comparison(self, original: np.ndarray,
                                  received: np.ndarray,
                                  sample_rate: int) -> plt.Figure:
        """
        Plot original vs received waveforms.
        
        Args:
            original: Original audio samples
            received: Received audio samples
            sample_rate: Sample rate in Hz
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Create time axis
        time_axis = np.arange(len(original)) / sample_rate
        
        # Plot original
        ax1.plot(time_axis, original, 
                color=self.colors['original'],
                alpha=0.8,
                linewidth=0.5)
        ax1.fill_between(time_axis, 0, original,
                        color=self.colors['original'],
                        alpha=0.3)
        ax1.set_title('Original Audio', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Amplitude', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, time_axis[-1]])
        
        # Plot received
        ax2.plot(time_axis, received,
                color=self.colors['received'],
                alpha=0.8,
                linewidth=0.5)
        ax2.fill_between(time_axis, 0, received,
                        color=self.colors['received'],
                        alpha=0.3)
        ax2.set_title('Received Audio', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time (seconds)', fontsize=12)
        ax2.set_ylabel('Amplitude', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, time_axis[-1]])
        
        plt.tight_layout()
        return fig
    
    def plot_packet_timeline(self, packets: List[RTPPacket],
                            frame_size_ms: int,
                            playout_delay_ms: int,
                            max_buffer_ms: int) -> plt.Figure:
        """
        Plot packet transmission timeline.
        
        Args:
            packets: List of packets
            frame_size_ms: Frame duration
            playout_delay_ms: Playout delay
            max_buffer_ms: Maximum buffer time
            
        Returns:
            Matplotlib figure
        """
        if not packets:
            return self._create_empty_plot("No packets to display")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Extract data
        seq_nums = [pkt.sequence for pkt in packets]
        send_times = [pkt.send_time * 1000 for pkt in packets]  # ms
        arrival_times = [pkt.arrival_time * 1000 for pkt in packets]  # ms
        
        # Calculate deadlines
        first_arrival = min(arrival_times)
        playout_start = first_arrival + playout_delay_ms
        deadline = playout_start + max_buffer_ms
        
        # Plot packet journeys
        for send, arrive, seq in zip(send_times, arrival_times, seq_nums):
            # Color based on arrival time
            if arrive <= playout_start + max_buffer_ms:
                color = self.colors['on_time']
                marker = 'o'
                size = 40
            else:
                color = self.colors['late']
                marker = 's'
                size = 50
            
            # Plot send and arrival points
            ax.scatter(send, seq, color='gray', alpha=0.5, s=20, marker='^')
            ax.scatter(arrive, seq, color=color, alpha=0.7, s=size, marker=marker)
            
            # Draw connecting line
            ax.plot([send, arrive], [seq, seq], 'gray', alpha=0.3, linewidth=1)
        
        # Draw reference lines
        ax.axvline(x=playout_start, color='green', linestyle='--',
                  linewidth=2, label='Playout Start')
        ax.axvline(x=deadline, color='red', linestyle='--',
                  linewidth=2, label='Late Deadline')
        
        # Draw expected playout times
        for seq in seq_nums:
            expected_time = playout_start + (seq * frame_size_ms)
            ax.axvline(x=expected_time, color='blue',
                      alpha=0.1, linestyle=':', linewidth=0.5)
        
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('Packet Sequence Number', fontsize=12)
        ax.set_title('Packet Transmission Timeline', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_delay_histogram(self, packets: List[RTPPacket]) -> plt.Figure:
        """
        Plot histogram of packet delays.
        
        Args:
            packets: List of packets
            
        Returns:
            Matplotlib figure
        """
        if not packets:
            return self._create_empty_plot("No packets to display")
        
        # Calculate delays in ms
        delays = []
        for pkt in packets:
            if pkt.send_time is not None and pkt.arrival_time is not None:
                delay = (pkt.arrival_time - pkt.send_time) * 1000
                delays.append(delay)
        
        if not delays:
            return self._create_empty_plot("No delay data available")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram
        ax1.hist(delays, bins=30, color=self.colors['original'],
                alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Delay (ms)', fontsize=12)
        ax1.set_ylabel('Number of Packets', fontsize=12)
        ax1.set_title('Packet Delay Distribution', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add statistics
        mean_delay = np.mean(delays)
        std_delay = np.std(delays)
        median_delay = np.median(delays)
        
        ax1.axvline(mean_delay, color='red', linestyle='--',
                   label=f'Mean: {mean_delay:.1f}ms')
        ax1.axvline(median_delay, color='green', linestyle=':',
                   label=f'Median: {median_delay:.1f}ms')
        ax1.legend()
        
        # Cumulative distribution
        sorted_delays = np.sort(delays)
        cdf = np.arange(1, len(sorted_delays) + 1) / len(sorted_delays)
        
        ax2.plot(sorted_delays, cdf, color=self.colors['received'],
                linewidth=2)
        ax2.set_xlabel('Delay (ms)', fontsize=12)
        ax2.set_ylabel('Cumulative Probability', fontsize=12)
        ax2.set_title('Delay Cumulative Distribution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add percentile lines
        for percentile in [50, 75, 90, 95, 99]:
            delay_value = np.percentile(delays, percentile)
            ax2.axvline(delay_value, color='gray', linestyle='--', alpha=0.5)
            ax2.text(delay_value, 0.05, f'{percentile}%',
                    rotation=90, va='bottom', ha='right')
        
        plt.tight_layout()
        return fig
    
    def plot_packet_statistics(self, stats: dict) -> plt.Figure:
        """
        Plot packet statistics pie chart and bar chart.
        
        Args:
            stats: Packet statistics dictionary
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Pie chart
        labels = ['On Time', 'Late', 'Lost']
        sizes = [
            stats.get('on_time_rate', 0) * 100,
            stats.get('late_rate', 0) * 100,
            stats.get('loss_rate', 0) * 100
        ]
        
        colors = [self.colors['on_time'], self.colors['late'], self.colors['lost']]
        
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, explode=(0.1, 0, 0))
        ax1.set_title('Packet Delivery Status', fontsize=14, fontweight='bold')
        
        # Bar chart with counts
        categories = ['Total Expected', 'On Time', 'Late', 'Lost']
        counts = [
            stats.get('total_expected', 0),
            stats.get('received_on_time', 0),
            stats.get('received_late', 0),
            stats.get('lost', 0)
        ]
        
        bar_colors = ['gray', self.colors['on_time'],
                     self.colors['late'], self.colors['lost']]
        
        bars = ax2.bar(categories, counts, color=bar_colors, alpha=0.8)
        ax2.set_title('Packet Counts', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Number of Packets', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def _create_empty_plot(self, message: str) -> plt.Figure:
        """Create an empty plot with a message."""
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, message,
                ha='center', va='center',
                fontsize=14, color='gray',
                transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        return fig