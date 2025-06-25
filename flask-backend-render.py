import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
from scipy.signal import butter, filtfilt, find_peaks
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'central_apnea_results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# --- CONFIG ---
fs = 25.0  # Sampling rate (Hz)
interval_duration = 60  # Duration per chunk (seconds)
samples_per_chunk = int(fs * interval_duration)

# --- FILTER ---
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(signal, lowcut=0.1, highcut=2.0):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return filtfilt(b, a, signal)

# --- DETECTION ---
def detect_breaths(sig, fs, prom=0.01):
    peaks, _ = find_peaks(sig, distance=int(fs * 0.5), prominence=prom)
    return peaks

def detect_central_apnea(t, sig, peaks, fs, win=8, amp_thr=0.1):
    events = []
    for i in range(1, len(peaks)):
        s, e = peaks[i - 1], peaks[i]
        seg = sig[s:e]
        duration = t[e] - t[s]
        amplitude = np.mean(np.abs(seg))
        if duration >= win and amplitude < amp_thr:
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

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        df = pd.read_csv(filepath)
        t = df['Time'].to_numpy()
        z = df['Z_g'].to_numpy()

        total_samples = len(z)
        num_chunks = total_samples // samples_per_chunk
        all_events = []
        encoded_plots = []

        for i in range(num_chunks):
            start = i * samples_per_chunk
            end = start + samples_per_chunk
            chunk_t = t[start:end]
            chunk_z = z[start:end]

            if len(chunk_t) < samples_per_chunk:
                continue

            filtered = apply_bandpass_filter(chunk_z)
            peaks = detect_breaths(filtered, fs)
            cen_events = detect_central_apnea(chunk_t, filtered, peaks, fs)
            cen_merged = merge_events(cen_events)

            if not cen_merged:
                continue

            # Plot
            plt.figure(figsize=(12, 6))
            plt.plot(chunk_t, chunk_z, label='Raw Z-axis', alpha=0.5)
            plt.plot(chunk_t, filtered, label='Filtered', color='orange')
            plt.plot(chunk_t[peaks], filtered[peaks], 'ro', label='Breath Peaks')
            for s, e in cen_merged:
                plt.axvspan(s, e, color='blue', alpha=0.2, label='Central Apnea')
            plt.title(f"Chunk {i + 1}: Central Apnea Detected")
            plt.xlabel("Time (s)")
            plt.ylabel("Z-axis Acceleration (g)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            encoded = base64.b64encode(buf.read()).decode('utf-8')
            encoded_plots.append(encoded)
            buf.close()
            plt.close()

            for s, e in cen_merged:
                all_events.append({'start_time': s, 'end_time': e, 'chunk': i+1})

        return jsonify({
            'central_apnea_events': all_events,
            'total_chunks_analyzed': num_chunks,
            'total_events': len(all_events),
            'plots': encoded_plots
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
