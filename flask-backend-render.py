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

# ---------- FILTER FUNCTIONS ----------
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    return butter(order, [lowcut / nyq, highcut / nyq], btype='band')

def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    return butter(order, cutoff / nyq, btype='low')

def apply_breathing_filter(signal):
    b, a = butter_bandpass(0.1, 0.4, FS)
    return filtfilt(b, a, signal)

def apply_central_filter(signal):
    b, a = butter_lowpass(0.1, FS)
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

def detect_breaths(t, sig, fs, mask, min_interval=3, prom=0.01, min_amp=0.001):
    min_distance = int(fs * min_interval)
    peaks, _ = find_peaks(sig, distance=min_distance, prominence=prom)
    valid_peaks = []

    for i, p in enumerate(peaks):
        if mask[p]:
            continue
        amp = sig[p]
        if abs(amp) < min_amp:
            continue
        prev_amp = sig[p - 1] if p > 0 else amp
        next_amp = sig[p + 1] if p < len(sig) - 1 else amp
        if amp > max(prev_amp, next_amp) * 0.2:
            valid_peaks.append(p)

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

def detect_central(t, sig, peaks, fs, win=8.0, amp_thr=0.004, uniformity_thr=0.2):
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

        seg_peaks, _ = find_peaks(seg, prominence=0.0005, distance=int(fs * 1.5))
        seg_times = t[s:e][seg_peaks]

        if len(seg_times) >= 3:
            intervals = np.diff(seg_times)
            mean_ivl = np.mean(intervals)
            std_ivl = np.std(intervals)
            if std_ivl > uniformity_thr * mean_ivl:
                continue

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
        'version': '1.0.0'
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

        # Load and validate CSV
        try:
            df = pd.read_csv(filepath)
            
            # Handle both comma and tab separated files
            if len(df.columns) == 1:
                df = pd.read_csv(filepath, sep='\t')
            
            if 'Time' not in df.columns or 'Z_g' not in df.columns:
                available_cols = ', '.join(df.columns.tolist())
                return jsonify({
                    'error': f'CSV must contain Time and Z_g columns. Found columns: {available_cols}'
                }), 400
                
            if not np.issubdtype(df['Time'].dtype, np.number):
                return jsonify({'error': 'Time column must contain numeric values (seconds)'}), 400
                
        except Exception as e:
            return jsonify({'error': f'Invalid CSV file: {str(e)}'}), 400

        t = df['Time'].values
        raw = df['Z_g'].values * AMPLIFICATION

        # Process signals
        unamplified_breathing = apply_breathing_filter(raw)
        breathing_signal = unamplified_breathing * 30
        unamplified_central = apply_central_filter(raw)
        obs_events, obs_mask = detect_obstructive(t, unamplified_breathing, FS)
        peaks = detect_breaths(t, unamplified_breathing, FS, obs_mask)
        cen_events = detect_central(t, unamplified_central, peaks, FS)

        obs_merged = merge_events(obs_events)
        cen_merged = merge_events(cen_events)

        # Save events to CSV
        events = []
        for s, e in obs_merged:
            events.append({'start_time': s, 'end_time': e, 'event': 'Obstructive Apnea'})
        for s, e in cen_merged:
            events.append({'start_time': s, 'end_time': e, 'event': 'Central Apnea'})
        
        events_df = pd.DataFrame(events)
        events_filename = os.path.splitext(filename)[0] + '_events.csv'
        events_path = os.path.join(ANALYSIS_FOLDER, events_filename)
        events_df.to_csv(events_path, index=False)

        # Create plot with better styling for web display
        plt.style.use('default')
        plt.figure(figsize=(14, 8))
        plt.axhline(y=0.15, color='red', linestyle='--', alpha=0.7, linewidth=1)
        plt.plot(t, raw, label='Raw Z-axis', alpha=0.6, color='blue', linewidth=0.8)
        plt.plot(t, breathing_signal, label='Breathing Signal (Filtered)', color='violet', linewidth=1.2)
        plt.plot(t[peaks], breathing_signal[peaks], 'ro', label='Breath Peaks', markersize=4)
        
        # Plot apnea events
        for i, (s, e) in enumerate(cen_merged):
            plt.axvspan(s, e, color='blue', alpha=0.2, 
                       label='Central Apnea' if i == 0 else "")
        for i, (s, e) in enumerate(obs_merged):
            plt.axvspan(s, e, color='orange', alpha=0.3, 
                       label='Obstructive Apnea' if i == 0 else "")
        
        plt.title("Sleep Apnea Detection Analysis", fontsize=16, fontweight='bold')
        plt.xlabel("Time (seconds)", fontsize=12)
        plt.ylabel("Z-axis Acceleration (g)", fontsize=12)
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot as base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close()

        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass

        return jsonify({
            'events_file': events_filename,
            'plot': plot_base64,
            'breath_count': len(peaks),
            'central_apnea_count': len(cen_merged),
            'obstructive_apnea_count': len(obs_merged),
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
