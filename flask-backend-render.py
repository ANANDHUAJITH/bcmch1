import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_from_directory
from scipy.signal import butter, filtfilt, find_peaks
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Use environment variables for production
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
ANALYSIS_FOLDER = os.environ.get('ANALYSIS_FOLDER', 'analysis_results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANALYSIS_FOLDER, exist_ok=True)

# ---------- CONFIG ----------
FS = 25.0  # Sampling frequency (Hz)
AMPLIFICATION = 2.0  # Amplification factor
CHUNK_DURATION = 60  # seconds
SAMPLES_PER_CHUNK = int(FS * CHUNK_DURATION)

# ---------- FILTER FUNCTIONS ----------
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    return butter(order, [lowcut / nyq, highcut / nyq], btype='band')

def apply_bandpass_filter(signal, lowcut=0.1, highcut=2.0):
    b, a = butter_bandpass(lowcut, highcut, FS)
    return filtfilt(b, a, signal)

# ---------- DETECTION FUNCTIONS ----------
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

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'Sleep Apnea Detection API is running',
        'version': '2.0.0'
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        filename = file.filename
        if not filename or not filename.endswith('.csv'):
            return jsonify({'error': 'File must be a CSV'}), 400

        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            df = pd.read_csv(filepath)
            if len(df.columns) == 1:
                df = pd.read_csv(filepath, sep='\t')

            if 'Time' not in df.columns or 'Z_g' not in df.columns:
                return jsonify({'error': 'CSV must contain Time and Z_g columns'}), 400

            if not np.issubdtype(df['Time'].dtype, np.number):
                return jsonify({'error': 'Time column must contain numeric values'}), 400

        except Exception as e:
            return jsonify({'error': f'Invalid CSV file: {str(e)}'}), 400

        t = df['Time'].values
        z = df['Z_g'].values * AMPLIFICATION

        total_samples = len(z)
        num_chunks = total_samples // SAMPLES_PER_CHUNK
        all_events = []
        all_peaks = []
        chunk_plots = []

        for i in range(num_chunks):
            start = i * SAMPLES_PER_CHUNK
            end = start + SAMPLES_PER_CHUNK
            chunk_t = t[start:end]
            chunk_z = z[start:end]

            if len(chunk_t) < SAMPLES_PER_CHUNK:
                continue

            filtered = apply_bandpass_filter(chunk_z)
            peaks = detect_breaths(filtered, FS)
            cen_events = detect_central_apnea(chunk_t, filtered, peaks, FS)
            cen_merged = merge_events(cen_events)

            all_peaks.extend([start + p for p in peaks])
            all_events.extend([(start + int(FS * (s - chunk_t[0])), start + int(FS * (e - chunk_t[0])), s, e) for s, e in cen_merged])

        # Build event summary
        events = [{'start_time': s, 'end_time': e, 'event': 'Central Apnea'} for _, _, s, e in all_events]
        events_df = pd.DataFrame(events)
        events_filename = os.path.splitext(filename)[0] + '_events.csv'
        events_path = os.path.join(ANALYSIS_FOLDER, events_filename)
        events_df.to_csv(events_path, index=False)

        # Plot full signal
        plt.figure(figsize=(14, 6))
        plt.plot(t, z, label='Raw Z-axis', alpha=0.5)
        filtered = apply_bandpass_filter(z)
        plt.plot(t, filtered, label='Filtered Signal', color='orange')
        plt.plot(t[all_peaks], filtered[all_peaks], 'ro', label='Breath Peaks')

        for i, (_, _, s, e) in enumerate(all_events):
            plt.axvspan(s, e, color='blue', alpha=0.2, label='Central Apnea' if i == 0 else '')

        plt.title("Sleep Apnea Detection")
        plt.xlabel("Time (s)")
        plt.ylabel("Z-axis Acceleration (g)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close()

        try:
            os.remove(filepath)
        except:
            pass

        return jsonify({
            'events_file': events_filename,
            'plot': plot_base64,
            'breath_count': len(all_peaks),
            'central_apnea_count': len(events),
            'total_duration': float(t[-1] - t[0]) if len(t) > 0 else 0,
            'data_points': len(t)
        })

    except Exception as e:
        print(f"Analysis error: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/download/<filename>', methods=['GET'])
def download(filename):
    try:
        return send_from_directory(ANALYSIS_FOLDER, os.path.basename(filename))
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
