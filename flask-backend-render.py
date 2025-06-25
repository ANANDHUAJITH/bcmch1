from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

app = Flask(__name__)
CORS(app)

FS = 10  # Sampling frequency (Hz)

# 🧠 Bandpass filter (0.1 - 0.8 Hz for respiration)
def apply_bandpass_filter(data, fs=FS, lowcut=0.1, highcut=0.8, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# ✅ Breath detection from filtered signal
def detect_breaths(filtered_signal, fs=FS, min_breath_interval=2.5):
    min_distance = int(min_breath_interval * fs)
    peaks, _ = find_peaks(filtered_signal, distance=min_distance, prominence=0.02)
    return peaks

# ✅ Central apnea detection based on breath intervals
def detect_central_apnea(time, filtered_signal, peaks, fs=FS):
    apnea_events = []
    min_apnea_duration = 10  # seconds
    min_apnea_samples = int(min_apnea_duration * fs)
    
    for i in range(1, len(peaks)):
        start = peaks[i - 1]
        end = peaks[i]
        duration = time[end] - time[start]
        
        if duration >= min_apnea_duration:
            event_time = (time[start] + time[end]) / 2
            apnea_events.append(round(event_time, 2))
    
    return apnea_events

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['file']
    df = pd.read_csv(file)

    # Make sure correct columns are used
    t = df['Time'].values
    raw = df['Z_g'].values  # g-force Z-axis values

    # Step-by-step detection pipeline
    filtered = apply_bandpass_filter(raw, fs=FS)
    peaks = detect_breaths(filtered, fs=FS)
    cen_events = detect_central_apnea(t, filtered, peaks, fs=FS)

    print("Central Apnea Events Detected:", cen_events)  # for debugging in backend

    return jsonify({
        'central_apnea_events': cen_events,
        'num_central_apnea_events': len(cen_events)
    })

if __name__ == '__main__':
    app.run(debug=True)
