
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

class EEGTimeGraphGenerator:
    '''
    Comprehensive EEG/Time graph generator for motor imagery BCI analysis

    This class provides multiple methods for generating EEG time-domain visualizations
    including single-channel plots, multi-channel plots, and interactive visualizations.
    '''

    def __init__(self, sampling_rate=1000, channel_names=None):
        self.sampling_rate = sampling_rate
        self.channel_names = channel_names or []

    def plot_single_channel_timeseries(self, data, channel_idx=0, title="EEG Time Series", 
                                     duration=None, start_time=0, figsize=(12, 6)):
        '''
        Plot single channel EEG time series

        Parameters:
        -----------
        data : numpy array
            EEG data with shape (n_channels, n_samples) or (n_samples,)
        channel_idx : int
            Index of channel to plot
        title : str
            Title for the plot
        duration : float
            Duration in seconds to plot (None for full signal)
        start_time : float
            Start time in seconds
        figsize : tuple
            Figure size (width, height)
        '''

        # Handle different data shapes
        if data.ndim == 1:
            signal_data = data
            channel_name = f"Channel {channel_idx}"
        else:
            signal_data = data[channel_idx, :]
            channel_name = self.channel_names[channel_idx] if channel_idx < len(self.channel_names) else f"Channel {channel_idx}"

        # Create time vector
        n_samples = len(signal_data)
        time_vector = np.arange(n_samples) / self.sampling_rate

        # Apply time constraints
        start_idx = int(start_time * self.sampling_rate)
        if duration is not None:
            end_idx = int((start_time + duration) * self.sampling_rate)
            end_idx = min(end_idx, n_samples)
        else:
            end_idx = n_samples

        time_plot = time_vector[start_idx:end_idx]
        signal_plot = signal_data[start_idx:end_idx]

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(time_plot, signal_plot, linewidth=0.8, color='blue', alpha=0.8)
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Amplitude (μV)', fontsize=12)
        ax.set_title(f'{title} - {channel_name}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        return fig, ax

    def plot_multichannel_timeseries(self, data, channels=None, title="Multi-Channel EEG Time Series",
                                   duration=None, start_time=0, figsize=(14, 10), 
                                   channel_spacing=100, show_events=None):
        '''
        Plot multiple EEG channels in a stacked time series plot

        Parameters:
        -----------
        data : numpy array
            EEG data with shape (n_channels, n_samples)
        channels : list
            List of channel indices to plot (None for all channels)
        title : str
            Title for the plot
        duration : float
            Duration in seconds to plot (None for full signal)
        start_time : float
            Start time in seconds
        figsize : tuple
            Figure size (width, height)
        channel_spacing : float
            Vertical spacing between channels
        show_events : dict
            Dictionary with event markers {'times': [...], 'labels': [...]}
        '''

        if data.ndim == 1:
            data = data.reshape(1, -1)

        n_channels, n_samples = data.shape
        channels = channels or list(range(n_channels))

        # Create time vector
        time_vector = np.arange(n_samples) / self.sampling_rate

        # Apply time constraints
        start_idx = int(start_time * self.sampling_rate)
        if duration is not None:
            end_idx = int((start_time + duration) * self.sampling_rate)
            end_idx = min(end_idx, n_samples)
        else:
            end_idx = n_samples

        time_plot = time_vector[start_idx:end_idx]

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        colors = plt.cm.tab10(np.linspace(0, 1, len(channels)))

        for i, ch_idx in enumerate(channels):
            signal_data = data[ch_idx, start_idx:end_idx]
            # Offset each channel vertically
            offset = i * channel_spacing
            ax.plot(time_plot, signal_data + offset, 
                   color=colors[i], linewidth=0.8, alpha=0.8,
                   label=self.channel_names[ch_idx] if ch_idx < len(self.channel_names) else f'Ch {ch_idx}')

        # Add event markers if provided
        if show_events is not None:
            for event_time, event_label in zip(show_events['times'], show_events['labels']):
                if start_time <= event_time <= (start_time + (duration or float('inf'))):
                    ax.axvline(x=event_time, color='red', linestyle='--', alpha=0.7)
                    ax.text(event_time, ax.get_ylim()[1], event_label, 
                           rotation=90, verticalalignment='bottom', fontsize=10)

        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Channels (with offset)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        return fig, ax

    def plot_trial_comparison(self, data, trial_labels, channels=None, 
                            title="Trial Comparison - EEG Time Series", figsize=(15, 8)):
        '''
        Plot EEG trials with different colors for different conditions

        Parameters:
        -----------
        data : numpy array
            EEG data with shape (n_trials, n_channels, n_samples)
        trial_labels : list
            Labels for each trial (e.g., ['Left', 'Right', 'Up', 'Down'])
        channels : list
            List of channel indices to plot
        title : str
            Title for the plot
        figsize : tuple
            Figure size (width, height)
        '''

        n_trials, n_channels, n_samples = data.shape
        channels = channels or [0]  # Default to first channel if none specified

        # Create time vector
        time_vector = np.arange(n_samples) / self.sampling_rate

        # Get unique labels and assign colors
        unique_labels = list(set(trial_labels))
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        label_colors = {label: colors[i] for i, label in enumerate(unique_labels)}

        fig, axes = plt.subplots(len(channels), 1, figsize=figsize, 
                                subplot_kw={'sharex': True})
        if len(channels) == 1:
            axes = [axes]

        for ch_idx, channel in enumerate(channels):
            ax = axes[ch_idx]

            # Plot each trial
            for trial_idx in range(n_trials):
                trial_data = data[trial_idx, channel, :]
                label = trial_labels[trial_idx]
                color = label_colors[label]

                ax.plot(time_vector, trial_data, color=color, alpha=0.6, linewidth=0.8)

            # Add channel name
            channel_name = self.channel_names[channel] if channel < len(self.channel_names) else f'Channel {channel}'
            ax.set_ylabel(f'{channel_name}\nAmplitude (μV)', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # Add legend
        legend_handles = [plt.Line2D([0], [0], color=label_colors[label], linewidth=2, label=label) 
                         for label in unique_labels]
        axes[0].legend(handles=legend_handles, loc='upper right', fontsize=10)

        axes[-1].set_xlabel('Time (seconds)', fontsize=12)
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        return fig, axes

    def plot_averaged_timeseries(self, data, trial_labels, channels=None,
                               title="Averaged EEG Time Series by Condition", figsize=(12, 8)):
        '''
        Plot averaged EEG time series for each condition with confidence intervals

        Parameters:
        -----------
        data : numpy array
            EEG data with shape (n_trials, n_channels, n_samples)
        trial_labels : list
            Labels for each trial
        channels : list
            List of channel indices to plot
        title : str
            Title for the plot
        figsize : tuple
            Figure size (width, height)
        '''

        n_trials, n_channels, n_samples = data.shape
        channels = channels or list(range(min(4, n_channels)))  # Default to first 4 channels

        # Create time vector
        time_vector = np.arange(n_samples) / self.sampling_rate

        # Get unique labels
        unique_labels = list(set(trial_labels))
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))

        fig, axes = plt.subplots(len(channels), 1, figsize=figsize, 
                                subplot_kw={'sharex': True})
        if len(channels) == 1:
            axes = [axes]

        for ch_idx, channel in enumerate(channels):
            ax = axes[ch_idx]

            for label_idx, label in enumerate(unique_labels):
                # Get trials for this condition
                condition_trials = [i for i, l in enumerate(trial_labels) if l == label]
                condition_data = data[condition_trials, channel, :]

                # Calculate mean and standard error
                mean_signal = np.mean(condition_data, axis=0)
                std_signal = np.std(condition_data, axis=0)
                sem_signal = std_signal / np.sqrt(len(condition_trials))

                color = colors[label_idx]

                # Plot mean
                ax.plot(time_vector, mean_signal, color=color, linewidth=2, label=label)

                # Plot confidence interval
                ax.fill_between(time_vector, 
                               mean_signal - sem_signal,
                               mean_signal + sem_signal,
                               color=color, alpha=0.3)

            # Add channel name
            channel_name = self.channel_names[channel] if channel < len(self.channel_names) else f'Channel {channel}'
            ax.set_ylabel(f'{channel_name}\nAmplitude (μV)', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if ch_idx == 0:
                ax.legend(loc='upper right', fontsize=10)

        axes[-1].set_xlabel('Time (seconds)', fontsize=12)
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        return fig, axes

    def plot_eeg_heatmap(self, data, channels=None, title="EEG Activity Heatmap",
                        duration=None, start_time=0, figsize=(14, 8)):
        '''
        Plot EEG data as a heatmap (channels vs time)

        Parameters:
        -----------
        data : numpy array
            EEG data with shape (n_channels, n_samples)
        channels : list
            List of channel indices to plot
        title : str
            Title for the plot
        duration : float
            Duration in seconds to plot
        start_time : float
            Start time in seconds
        figsize : tuple
            Figure size (width, height)
        '''

        if data.ndim == 1:
            data = data.reshape(1, -1)

        n_channels, n_samples = data.shape
        channels = channels or list(range(n_channels))

        # Apply time constraints
        start_idx = int(start_time * self.sampling_rate)
        if duration is not None:
            end_idx = int((start_time + duration) * self.sampling_rate)
            end_idx = min(end_idx, n_samples)
        else:
            end_idx = n_samples

        # Extract data for plotting
        plot_data = data[channels, start_idx:end_idx]

        # Create time labels
        time_vector = np.arange(start_idx, end_idx) / self.sampling_rate

        # Create channel labels
        channel_labels = [self.channel_names[ch] if ch < len(self.channel_names) else f'Ch {ch}' 
                         for ch in channels]

        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)

        # Downsample for visualization if data is too large
        max_time_points = 1000
        if plot_data.shape[1] > max_time_points:
            step = plot_data.shape[1] // max_time_points
            plot_data = plot_data[:, ::step]
            time_vector = time_vector[::step]

        im = ax.imshow(plot_data, aspect='auto', cmap='RdYlBu_r', 
                      interpolation='nearest')

        # Set labels
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Channels', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Set ticks
        n_time_ticks = min(10, len(time_vector))
        time_tick_idx = np.linspace(0, len(time_vector)-1, n_time_ticks, dtype=int)
        ax.set_xticks(time_tick_idx)
        ax.set_xticklabels([f'{time_vector[i]:.1f}' for i in time_tick_idx])

        ax.set_yticks(range(len(channels)))
        ax.set_yticklabels(channel_labels)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Amplitude (μV)', fontsize=12)

        plt.tight_layout()
        return fig, ax

    def plot_interactive_timeseries(self, data, channels=None, 
                                  title="Interactive EEG Time Series", figsize=(15, 10)):
        '''
        Create an interactive plot with zoom and pan capabilities

        Parameters:
        -----------
        data : numpy array
            EEG data with shape (n_channels, n_samples)
        channels : list
            List of channel indices to plot
        title : str
            Title for the plot
        figsize : tuple
            Figure size (width, height)
        '''

        if data.ndim == 1:
            data = data.reshape(1, -1)

        n_channels, n_samples = data.shape
        channels = channels or list(range(min(8, n_channels)))  # Limit to 8 channels for readability

        # Create time vector
        time_vector = np.arange(n_samples) / self.sampling_rate

        # Create subplots
        fig, axes = plt.subplots(len(channels), 1, figsize=figsize, 
                                subplot_kw={'sharex': True})
        if len(channels) == 1:
            axes = [axes]

        for i, ch_idx in enumerate(channels):
            signal_data = data[ch_idx, :]
            channel_name = self.channel_names[ch_idx] if ch_idx < len(self.channel_names) else f'Channel {ch_idx}'

            axes[i].plot(time_vector, signal_data, linewidth=0.8, color='blue')
            axes[i].set_ylabel(f'{channel_name}\n(μV)', fontsize=10)
            axes[i].grid(True, alpha=0.3)
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)

        axes[-1].set_xlabel('Time (seconds)', fontsize=12)
        fig.suptitle(title, fontsize=14, fontweight='bold')

        # Enable interactive navigation
        plt.subplots_adjust(hspace=0.3)

        return fig, axes

# Example usage functions
def generate_sample_eeg_data(n_channels=8, n_samples=5000, n_trials=20, sampling_rate=1000):
    '''Generate sample EEG data for demonstration'''
    np.random.seed(42)

    # Single trial data
    single_data = np.random.randn(n_channels, n_samples) * 50

    # Add some realistic EEG-like oscillations
    time_vector = np.arange(n_samples) / sampling_rate
    for ch in range(n_channels):
        # Add alpha rhythm (8-12 Hz)
        alpha_freq = 8 + np.random.randn() * 2
        single_data[ch, :] += 30 * np.sin(2 * np.pi * alpha_freq * time_vector)

        # Add some beta activity (13-30 Hz)
        beta_freq = 13 + np.random.randn() * 8
        single_data[ch, :] += 15 * np.sin(2 * np.pi * beta_freq * time_vector)

    # Multi-trial data
    multi_data = np.random.randn(n_trials, n_channels, n_samples) * 50
    trial_labels = ['Left', 'Right', 'Up', 'Down'] * (n_trials // 4)

    return single_data, multi_data, trial_labels

def demo_eeg_plotting():
    '''Demonstrate EEG time graph generation'''

    # Generate sample data
    single_data, multi_data, trial_labels = generate_sample_eeg_data()

    # Channel names for demonstration
    channel_names = ['Fp1', 'Fp2', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']

    # Initialize the EEG time graph generator
    eeg_plotter = EEGTimeGraphGenerator(sampling_rate=1000, channel_names=channel_names)

    print("Generating EEG time graphs...")

    # 1. Single channel time series
    fig1, ax1 = eeg_plotter.plot_single_channel_timeseries(
        single_data, channel_idx=2, title="Single Channel EEG", duration=3.0
    )
    plt.savefig('single_channel_eeg.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2. Multi-channel time series
    fig2, ax2 = eeg_plotter.plot_multichannel_timeseries(
        single_data, channels=[0, 2, 4, 6], title="Multi-Channel EEG", 
        duration=5.0, channel_spacing=100
    )
    plt.savefig('multichannel_eeg.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 3. Trial comparison
    fig3, axes3 = eeg_plotter.plot_trial_comparison(
        multi_data, trial_labels, channels=[2, 3], 
        title="Motor Imagery Trial Comparison"
    )
    plt.savefig('trial_comparison_eeg.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 4. Averaged time series
    fig4, axes4 = eeg_plotter.plot_averaged_timeseries(
        multi_data, trial_labels, channels=[2, 3, 4, 5],
        title="Averaged EEG by Motor Imagery Condition"
    )
    plt.savefig('averaged_eeg.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 5. EEG heatmap
    fig5, ax5 = eeg_plotter.plot_eeg_heatmap(
        single_data, channels=list(range(8)), 
        title="EEG Activity Heatmap", duration=10.0
    )
    plt.savefig('eeg_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("All EEG time graphs generated successfully!")
    return eeg_plotter

if __name__ == "__main__":
    # Run the demo
    plotter = demo_eeg_plotting()
