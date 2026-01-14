```python
import numpy as np
import mne
from scipy.io import loadmat
from scipy.signal import welch
import matplotlib.pyplot as plt
from MFDFA import MFDFA
import glob

# 1. Load and Preprocess Data
def load_bci_data(file_path):
    """Load a single .mat file and return BCI structure."""
    mat = loadmat(file_path)
    bci = mat['BCI']
    if isinstance(bci, np.ndarray) and bci.shape == (1, 1):
        bci = bci[0, 0]
    return bci

def create_mne_raw(bci, trial_idx=4):
    """Create MNE Raw object for a single trial."""
    # Get channel names
    try:
        if hasattr(bci, 'dtype') and bci.dtype.names and 'chaninfo' in bci.dtype.names:
            chaninfo = bci['chaninfo']
            if chaninfo.dtype.names and 'label' in chaninfo.dtype.names:
                label_field = chaninfo[0, 0]['label']
                ch_names = [label.item() if isinstance(label, np.ndarray) else label for label in label_field.flatten()]
                if not all(isinstance(name, str) for name in ch_names):
                    raise ValueError("Not all channel names are strings")
            else:
                raise KeyError("label field not found in chaninfo")
        else:
            raise KeyError("chaninfo field not found in BCI")
    except (IndexError, KeyError, ValueError) as e:
        print(f"Error accessing chaninfo.label: {e}")
        print("BCI dtype.names:", bci.dtype.names if hasattr(bci, 'dtype') else "N/A")
        print("chaninfo content:", bci['chaninfo'])
        raise
    
    # Get EEG data
    try:
        # Try accessing data as 2D array
        eeg_data = bci['data'][0, trial_idx]
        if eeg_data.ndim == 1:
            # If 1D, check if it can be reshaped based on n_channels
            n_channels = len(ch_names)
            n_samples = eeg_data.size // n_channels
            if n_samples * n_channels == eeg_data.size:
                eeg_data = eeg_data.reshape(n_channels, n_samples)
            else:
                raise ValueError(f"Cannot reshape 1D data of size {eeg_data.size} into (n_channels={n_channels}, n_samples)")
        elif eeg_data.ndim == 3 and eeg_data.shape[0] == 1:
            eeg_data = eeg_data[0]  # Remove singleton dimension
        if eeg_data.ndim != 2 or eeg_data.shape[0] != len(ch_names):
            raise ValueError(f"EEG data shape {eeg_data.shape} does not match n_channels={len(ch_names)}")
    except (IndexError, KeyError, ValueError) as e:
        print(f"Error accessing EEG data: {e}")
        print("bci['data'] shape:", bci['data'].shape)
        print("bci['data'][0, trial_idx] shape:", bci['data'][0, trial_idx].shape if trial_idx < bci['data'].shape[1] else "Out of bounds")
        raise
    
    sfreq = bci['SRATE'][0, 0]  # 1000 Hz
    
    # Create MNE Info object
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    
    # Create Raw object
    raw = mne.io.RawArray(eeg_data * 1e-6, info)  # Convert μV to V for MNE
    raw.set_montage('standard_1020')  # Use standard 10-10 montage
    
    # Filter for alpha band (8-14 Hz)
    raw.filter(8, 14, fir_design='firwin')
    
    # Exclude noisy channels
    try:
        noisy_channels = bci['chaninfo'][0, 0]['noisechan']
        if len(noisy_channels) > 0:
            bad_channels = [ch_names[i] for i in noisy_channels]
            raw.info['bads'] = bad_channels
            raw.interpolate_bads()
    except (IndexError, KeyError):
        print("No noisy channels found or error accessing noisechan.")
    
    return raw

# 2. Visualizations
def plot_time_series(raw, channels=['C3', 'C4']):
    """Plot EEG time series for selected channels."""
    try:
        raw.plot(scalings='auto', n_channels=2, show=False, block=False, picks=channels)
        plt.title('EEG Time Series (C3, C4)')
        plt.savefig('eeg_time_series.png')
        plt.close()
    except ValueError as e:
        print(f"Error plotting time series: {e}. Check if channels {channels} exist.")

def plot_psd(raw, channels=['C3', 'C4']):
    """Plot PSD for selected channels."""
    try:
        raw.compute_psd(fmin=8, fmax=14, picks=channels).plot(show=False)
        plt.title('Power Spectral Density (8-14 Hz)')
        plt.savefig('psd_c3_c4.png')
        plt.close()
    except ValueError as e:
        print(f"Error plotting PSD: {e}. Check if channels {channels} exist.")

def plot_erd_ers(bci, trial_idx=4):
    """Plot ERD/ERS for C3/C4 during a trial."""
    try:
        label_field = bci['chaninfo'][0, 0]['label']
        ch_names = [label.item() if isinstance(label, np.ndarray) else label for label in label_field.flatten()]
        c3_idx = ch_names.index('C3')
        c4_idx = ch_names.index('C4')
        eeg_data = bci['data'][0, trial_idx]
        if eeg_data.ndim == 1:
            n_channels = len(ch_names)
            n_samples = eeg_data.size // n_channels
            if n_samples * n_channels == eeg_data.size:
                eeg_data = eeg_data.reshape(n_channels, n_samples)
            else:
                raise ValueError(f"Cannot reshape 1D data of size {eeg_data.size} into (n_channels={n_channels}, n_samples)")
        elif eeg_data.ndim == 3 and eeg_data.shape[0] == 1:
            eeg_data = eeg_data[0]
        eeg_c3 = eeg_data[c3_idx, :]
        eeg_c4 = eeg_data[c4_idx, :]
    except (ValueError, KeyError, IndexError) as e:
        print(f"Error in plot_erd_ers: {e}")
        return
    
    sfreq = bci['SRATE'][0, 0]
    
    # Compute PSD for inter-trial (-2000 to 0 ms) and feedback (2000 ms to resultind)
    try:
        resultind = bci['TrialData'][0, trial_idx]['resultind'][0, 0]
    except (IndexError, KeyError):
        print(f"Error accessing TrialData.resultind for trial {trial_idx}")
        return
    
    inter_trial = slice(0, 2000)  # -2000 to 0 ms
    feedback = slice(4000, int(resultind))  # 2000 ms to end of feedback
    
    freqs, psd_c3_inter = welch(eeg_c3[inter_trial], fs=sfreq, nperseg=1024)
    _, psd_c3_feed = welch(eeg_c3[feedback], fs=sfreq, nperseg=1024)
    _, psd_c4_inter = welch(eeg_c4[inter_trial], fs=sfreq, nperseg=1024)
    _, psd_c4_feed = welch(eeg_c4[feedback], fs=sfreq, nperseg=1024)
    
    # Compute ERD: (feedback power - inter-trial power) / inter-trial power
    alpha_idx = (freqs >= 8) & (freqs <= 14)
    erd_c3 = (np.mean(psd_c3_feed[alpha_idx]) - np.mean(psd_c3_inter[alpha_idx])) / np.mean(psd_c3_inter[alpha_idx])
    erd_c4 = (np.mean(psd_c4_feed[alpha_idx]) - np.mean(psd_c4_inter[alpha_idx])) / np.mean(psd_c4_inter[alpha_idx])
    
    # Plot
    plt.figure(figsize=(6, 4))
    plt.bar(['C3', 'C4'], [erd_c3, erd_c4], color=['red', 'blue'])
    plt.ylabel('ERD/ERS (Relative Power Change)')
    plt.title(f'ERD/ERS for Trial {trial_idx + 1}')
    plt.savefig('erd_ers.png')
    plt.close()

def plot_topography(bci, trial_idx=4):
    """Plot topographic map of alpha power."""
    try:
        label_field = bci['chaninfo'][0, 0]['label']
        ch_names = [label.item() if isinstance(label, np.ndarray) else label for label in label_field.flatten()]
        eeg_data = bci['data'][0, trial_idx]
        if eeg_data.ndim == 1:
            n_channels = len(ch_names)
            n_samples = eeg_data.size // n_channels
            if n_samples * n_channels == eeg_data.size:
                eeg_data = eeg_data.reshape(n_channels, n_samples)
            else:
                raise ValueError(f"Cannot reshape 1D data of size {eeg_data.size} into (n_channels={n_channels}, n_samples)")
        elif eeg_data.ndim == 3 and eeg_data.shape[0] == 1:
            eeg_data = eeg_data[0]
    except (IndexError, KeyError, ValueError) as e:
        print(f"Error accessing EEG data for topography: {e}")
        return
    
    sfreq = bci['SRATE'][0, 0]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(eeg_data * 1e-6, info)
    raw.set_montage('standard_1020')
    raw.filter(8, 14)
    
    psd, freqs = mne.time_frequency.psd_welch(raw, fmin=8, fmax=14, n_per_seg=1024)
    alpha_power = psd.mean(axis=1)
    mne.viz.plot_topomap(alpha_power, raw.info, show=False)
    plt.title(f'Alpha Power Topography (Trial {trial_idx + 1})')
    plt.savefig('topography.png')
    plt.close()

def compute_pvc(bci):
    """Compute Percent Valid Correct (PVC)."""
    try:
        results = np.array([bci['TrialData'][0, i]['result'][0, 0] for i in range(bci['TrialData'].shape[1])])
        hits = np.sum(results == 1)
        misses = np.sum(results == 0)
        return hits / (hits + misses) * 100 if (hits + misses) > 0 else 0
    except (IndexError, KeyError) as e:
        print(f"Error computing PVC: {e}")
        return None

def plot_pvc_across_sessions(subject_id='S1'):
    """Plot PVC across sessions for a subject."""
    pvc_list = []
    session_files = sorted(glob.glob(f'{subject_id}_Session_*.mat'))
    if not session_files:
        print(f"No session files found for {subject_id}")
        return
    
    for file in session_files:
        bci = load_bci_data(file)
        pvc = compute_pvc(bci)
        if pvc is not None:
            pvc_list.append(pvc)
    
    print(f"PVC values for {subject_id}: {pvc_list}")
    if pvc_list:
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(pvc_list) + 1), pvc_list, marker='o', color='#1e88e5')
        plt.xlabel('Session')
        plt.ylabel('Percent Valid Correct (%)')
        plt.title(f'PVC Across Sessions for {subject_id}')
        plt.grid(True)
        plt.savefig('pvc_sessions.png')
        plt.close()
    else:
        print("No valid PVC data to plot.")

# 3. MFDFA Analysis
def perform_mfdfa(bci, channel='C3', trial_idx=4):
    """Perform MFDFA on a single channel's EEG data."""
    try:
        label_field = bci['chaninfo'][0, 0]['label']
        ch_names = [label.item() if isinstance(label, np.ndarray) else label for label in label_field.flatten()]
        ch_idx = ch_names.index(channel)
        eeg_data = bci['data'][0, trial_idx]
        if eeg_data.ndim == 1:
            n_channels = len(ch_names)
            n_samples = eeg_data.size // n_channels
            if n_samples * n_channels == eeg_data.size:
                eeg_data = eeg_data.reshape(n_channels, n_samples)
            else:
                raise ValueError(f"Cannot reshape 1D data of size {eeg_data.size} into (n_channels={n_channels}, n_samples)")
        elif eeg_data.ndim == 3 and eeg_data.shape[0] == 1:
            eeg_data = eeg_data[0]
        eeg_data = eeg_data[ch_idx, :]
    except (ValueError, KeyError, IndexError) as e:
        print(f"Error in perform_mfdfa: {e}")
        return None, None, None
    
    # MFDFA parameters
    scales = np.arange(16, 2048, 32)  # Scales for detrended fluctuation
    q = np.linspace(-5, 5, 101)  # q-orders for multifractal analysis
    
    # Compute MFDFA
    lag, dfa = MFDFA(eeg_data, scales, q)
    
    # Calculate Hurst exponent and singularity spectrum
    h_q = []
    for i in range(len(q)):
        coef = np.polyfit(np.log2(lag), np.log2(dfa[i, :]), 1)
        h_q.append(coef[0])
    
    # Singularity spectrum
    alpha = h_q + q * np.gradient(h_q, q[1] - q[0])
    f_alpha = q * alpha - (h_q * q - np.log2(dfa[:, -1]))
    
    # Plot multifractal spectrum
    plt.figure(figsize=(6, 4))
    plt.plot(alpha, f_alpha, 'b-')
    plt.xlabel('Singularity Strength (α)')
    plt.ylabel('Multifractal Spectrum f(α)')
    plt.title(f'MFDFA Spectrum for {channel} (Trial {trial_idx + 1})')
    plt.grid(True)
    plt.savefig('mfdfa_spectrum.png')
    plt.close()
    
    return alpha, f_alpha, h_q

# Main execution
if __name__ == '__main__':
    # Load sample .mat file
    file_path = 'S1_Session_1.mat'  # Replace with your file path
    try:
        bci = load_bci_data(file_path)
        
        # Inspect BCI structure
        print("BCI type:", type(bci))
        print("BCI shape:", bci.shape if isinstance(bci, np.ndarray) else "N/A")
        print("BCI dtype.names:", bci.dtype.names if hasattr(bci, 'dtype') else "N/A")
        print("BCI keys:", list(bci.keys()) if isinstance(bci, dict) else "Not a dict")
        if hasattr(bci, 'dtype') and bci.dtype.names and 'chaninfo' in bci.dtype.names:
            print("chaninfo content:", bci['chaninfo'])
            if bci['chaninfo'].dtype.names:
                print("chaninfo dtype.names:", bci['chaninfo'].dtype.names)
                print("chaninfo[0, 0]['label'] content:", bci['chaninfo'][0, 0]['label'])
                ch_names = [label.item() if isinstance(label, np.ndarray) else label for label in bci['chaninfo'][0, 0]['label'].flatten()]
                print("Extracted channel names:", ch_names)
        
        # Inspect data structure
        print("bci['data'] type:", type(bci['data']))
        print("bci['data'] shape:", bci['data'].shape)
        print("bci['data'][0, 4] type:", type(bci['data'][0, 4]))
        print("bci['data'][0, 4] shape:", bci['data'][0, 4].shape)
        print("bci['data'][0, 4][0] shape:", bci['data'][0, 4][0].shape)
        
        # Create MNE Raw object for trial 5
        raw = create_mne_raw(bci, trial_idx=4)
        
        # Generate visualizations
        plot_time_series(raw, channels=['C3', 'C4'])
        plot_psd(raw, channels=['C3', 'C4'])
        plot_erd_ers(bci, trial_idx=4)
        plot_topography(bci, trial_idx=4)
        plot_pvc_across_sessions(subject_id='S1')
        
        # Perform MFDFA
        alpha, f_alpha, h_q = perform_mfdfa(bci, channel='C3', trial_idx=4)
        if alpha is not None:
            print(f'Multifractal spectrum width: {max(alpha) - min(alpha):.2f}')
    except Exception as e:
        print(f"Error in main execution: {e}")
```

### Key Changes
1. **EEG Data Handling**:
   - In `create_mne_raw`, `plot_erd_ers`, `plot_topography`, and `perform_mfdfa`, changed `eeg_data = bci['data'][0, trial_idx][0]` to `eeg_data = bci['data'][0, trial_idx]`.
   - Added logic to handle 1D or 3D data:
     - If 1D, attempt to reshape to `(n_channels, n_samples)` using `len(ch_names)`.
     - If 3D with a singleton dimension, remove it with `eeg_data[0]`.
     - Validate that the final shape matches `(n_channels, n_samples)`.

2. **Error Checking**:
   - Added checks to ensure `eeg_data` shape is compatible with `ch_names`.
   - Print detailed error messages with `bci['data']` shapes if reshaping fails.

3. **Diagnostics**:
   - Added prints in the main block to show `bci['data']`, `bci['data'][0, 4]`, and `bci['data'][0, 4][0]` shapes.

### Expected Outputs
Running the code should:
- Print the structure of `bci['data']`:
  ```
  bci['data'] type: <class 'numpy.ndarray'>
  bci['data'] shape: (1, n_trials)
  bci['data'][0, 4] type: <class 'numpy.ndarray'>
  bci['data'][0, 4] shape: (62, 11041) or (1, 11041) or (n_channels, n_samples)
  bci['data'][0, 4][0] shape: (11041,)
  ```
- Generate plots if the data shape is corrected:
  - `eeg_time_series.png`: EEG signals for C3/C4.
  - `psd_c3_c4.png`: PSD for C3/C4 (8–14 Hz).
  - `erd_ers.png`: ERD/ERS bar plot for C3/C4.
  - `topography.png`: Alpha power scalp map.
  - `pvc_sessions.png`: PVC across sessions (if multiple files exist).
  - `mfdfa_spectrum.png`: MFDFA spectrum for C3.
- Print PVC values, e.g.:
  ```
  PVC values for S1: [60.0, 65.0, 70.0, ...]
  ```
- Print the multifractal spectrum width, e.g.:
  ```
  Multifractal spectrum width: 0.85
  ```

### Chart.js Visualization
For the PVC plot, use the actual values printed by `plot_pvc_across_sessions`. If you share the `PVC values for S1` output, I can update this chart:

```chartjs
{
  "type": "line",
  "data": {
    "labels": ["Session 1", "Session 2", "Session 3", "Session 4", "Session 5", "Session 6", "Session 7"],
    "datasets": [{
      "label": "PVC (%)",
      "data": [60, 65, 68, 70, 72, 75, 78],  // Replace with actual PVC values
      "borderColor": "#1e88e5",
      "backgroundColor": "rgba(30, 136, 229, 0.2)",
      "fill": true
    }]
  },
  "options": {
    "scales": {
      "y": {
        "beginAtZero": true,
        "title": { "display": true, "text": "Percent Valid Correct (%)" }
      },
      "x": {
        "title": { "display": true, "text": "Session" }
      }
    },
    "plugins": {
      "title": { "display": true, "text": "PVC Across Sessions for S1" }
    }
  }
}
```

### Troubleshooting and Next Steps
1. **Run the Updated Code**:
   - Run the provided code and share the console output, especially:
     - `bci['data']` type and shape.
     - `bci['data'][0, 4]` type and shape.
     - `bci['data'][0, 4][0]` shape.
     - `PVC values for S1`.
     - Any new error messages.

2. **Confirm Data Shape**:
   - The output will clarify if `bci['data'][0, trial_idx]` is `(62, 11041)` or another shape.
   - If `bci['data'][0, 4]` is 1D with size `62 * 11041 = 684542`, the reshaping logic should work (since `684542 / 62 ≈ 11041`).

3. **Check Trial Index**:
   - Ensure `trial_idx=4` is valid. If `bci['data'].shape[1] < 5`, try `trial_idx=0` and re-run.

4. **File Availability**:
   - Share how many session files you have (e.g., `S1_Session_1.mat`, `S1_Session_2.mat`, ...).
   - Confirm `file_path = 'S1_Session_1.mat'` is correct.

5. **Analysis Goals**:
   - Specify if you want to focus on specific tasks (LR, UD, 2D) using `bci['TrialData'][0, i]['tasknumber']` (1=LR, 2=UD, 3=2D).
   - Interested in MBSR vs. control groups? Check `bci['metadata'][0, 0]['MBSRsubject']`.
   - Want to analyze other channels (e.g., CZ, FC3) or adjust MFDFA parameters?

6. **Additional Visualizations**:
   - I can create Chart.js plots for ERD/ERS across trials, alpha power by channel, or task-specific metrics. Share your preferences.
   - If you provide PVC values or other metrics (e.g., Hurst exponents), I can tailor charts.

Please run the updated code and share:
- The console output, especially the `bci['data']` diagnostics and `PVC values for S1`.
- Any new errors.
- The number of session files.
- Your analysis goals (e.g., specific tasks, MBSR vs. control, additional channels).

This will help me confirm the data structure and provide further fixes or tailored analyses!