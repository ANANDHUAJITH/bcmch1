import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

from scipy.signal import find_peaks
import numpy as np
# ---------- CONFIG ----------
fs = 25.0  # Sampling frequency (Hz)
amplification = 2.0  # Amplification factor

# ---------- FILTER FUNCTIONS ----------
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    return butter(order, [lowcut / nyq, highcut / nyq], btype='band')

def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    return butter(order, cutoff / nyq, btype='low')

def apply_breathing_filter(signal):
    b, a = butter_bandpass(0.1, 0.4, fs)  # Violet signal for breathing
    return filtfilt(b, a, signal)

def apply_central_filter(signal):
    b, a = butter_lowpass(0.1, fs)  # Green signal for central apnea
    return filtfilt(b, a, signal)

# ---------- DETECTION FUNCTIONS ----------
def detect_obstructive(t, sig, fs, amp_thresh=0.05, zero_cross_thresh=2, win=8):
    mask = np.zeros(len(sig), dtype=bool)
    events = []
    w = int(fs * win)
    sig_demean = sig - np.mean(sig)

    for i in range(0, len(sig) - w, int(fs)):
        seg = sig_demean[i:i + w]
        amp = np.mean(np.abs(seg))
        zero_crossings = np.where(np.diff(np.sign(seg)))[0]
        zero_count = len(zero_crossings)

        if amp > amp_thresh and zero_count > zero_cross_thresh:
            events.append((t[i], t[i + w]))
            mask[i:i + w] = True
    return events, mask

def detect_breaths(t, sig, fs, mask, min_interval=2.5, prom=0.001, min_amp=0.001, debug=False):
    min_distance = int(fs * min_interval)
    peaks, properties = find_peaks(sig, distance=min_distance, prominence=prom)
    valid_peaks = []

    for i, p in enumerate(peaks):
        if mask[p]:
            if debug:
                print(f"Rejected @ {t[p]:.2f}s — Masked (apnea)")
            continue

        amp = sig[p]
        if abs(amp) < min_amp:
            if debug:
                print(f"Rejected @ {t[p]:.2f}s — Too small: {amp:.5f}")
            continue

        prev_amp = sig[p - 1] if p > 0 else amp
        next_amp = sig[p + 1] if p < len(sig) - 1 else amp
        if amp > max(prev_amp, next_amp) * 0.2:  # Corrected to detect local maxima
            valid_peaks.append(p)
        else:
            if debug:
                print(f"Rejected @ {t[p]:.2f}s — Not sharp enough")

    # Filter peaks for timing consistency
    filtered_peaks = []
    for i, p in enumerate(valid_peaks):
        if i == 0 or i == len(valid_peaks) - 1:
            filtered_peaks.append(p)
        else:
            prev_t = t[valid_peaks[i - 1]]
            curr_t = t[p]
            next_t = t[valid_peaks[i + 1]]
            interval_prev = curr_t - prev_t
            interval_next = next_t - curr_t
            if 2.5 <= interval_prev <= 6.0 or 2.5 <= interval_next <= 6.0:
                filtered_peaks.append(p)

    return filtered_peaks




def detect_central(t, sig, peaks, fs, win=10.0, amp_thr=0.004, uniformity_thr=0.2):
    events = []
    sig_demean = sig - np.mean(sig)

    for i in range(1, len(peaks)):
        s, e = peaks[i - 1], peaks[i]
        seg = sig_demean[s:e]
        duration = t[e] - t[s]

        if duration < win:
            continue

        amp = np.mean(np.abs(seg))
        if amp > amp_thr:
            continue

        # ---------- Internal peak detection within the segment ----------
        seg_peaks, _ = find_peaks(seg, prominence=0.0005, distance=int(fs * 1.5))
        seg_times = t[s:e][seg_peaks]

        # ---------- Uniformity check ----------
        if len(seg_times) >= 3:  # Need at least 3 to check uniformity
            intervals = np.diff(seg_times)
            mean_ivl = np.mean(intervals)
            std_ivl = np.std(intervals)

            if std_ivl > uniformity_thr * mean_ivl:
                continue  # Too irregular — not central apnea

        events.append((t[s], t[e]))

    return events


def merge_events(events, gap=2):
    if not events:
        return []
    merged = [events[0]]
    for s, e in events[1:]:
        if s - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], max(e, merged[-1][1]))
        else:
            merged.append((s, e))
    return merged

# ---------- LOAD CSV ----------
csv_file = r'C:\Users\91974\Downloads\z_axis_2 (1).csv'
df = pd.read_csv(csv_file)
df = df[(df['Time'] >=0.28) ]  # Only 250s to 320s
t = df['Time'].values
raw = df['Z_g'].values * amplification

# ---------- PROCESS ----------
unamplified_breathing = apply_breathing_filter(raw)  # Unamplified for detection
breathing_signal = unamplified_breathing * 30        # Amplify for readability
unamplified_central = apply_central_filter(raw)     # Unamplified for detection
central_signal = unamplified_central * 5            # Amplify for readability
obs_events, obs_mask = detect_obstructive(t, unamplified_breathing, fs)
peaks = detect_breaths(t, unamplified_breathing, fs, obs_mask)
cen_events = detect_central(t, unamplified_central, peaks, fs)

obs_merged = merge_events(obs_events)
cen_merged = merge_events(cen_events)

# ---------- PLOT ----------
plt.figure(figsize=(12, 6))
plt.axhline(y=0.15, color='red', linestyle='--')

plt.plot(t, raw, label='Raw Z-axis', alpha=0.5, color='blue')
plt.plot(t, breathing_signal, label='Breathing Signal (Filtered)', color='violet')
#plt.plot(t, central_signal, label='Central Apnea Signal (Filtered)', color='green')
plt.plot(t[peaks], breathing_signal[peaks], 'ro', label='Breath Peaks')

for s, e in cen_merged:
    plt.axvspan(s, e, color='blue', alpha=0.2, label='Central Apnea' if s == cen_merged[0][0] else "")
for s, e in obs_merged:
    plt.axvspan(s, e, color='orange', alpha=0.3, label='Obstructive Apnea' if s == obs_merged[0][0] else "")

plt.title("Sleep Apnea Detection (from CSV)")
plt.xlabel("Time (s)")
plt.ylabel("Z-axis Accel (g)")
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------- SUMMARY ----------
print(f"Total Breaths Detected: {len(peaks)}")
print(f"Central Apnea Events: {len(cen_merged)}")
print(f"Obstructive Apnea Events: {len(obs_merged)}")
