from flask import Flask, request, jsonify
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data)

def detect_breaths(sig, fs, prom=0.005):
    peaks, _ = find_peaks(sig, distance=int(fs * 0.5), prominence=prom)
    return peaks

def detect_central_apnea(t, sig, peaks, fs, win=8, amp_thr=0.03):
    events = []
    sig_demean = sig - np.mean(sig)
    for i in range(1, len(peaks)):
        s, e = peaks[i - 1], peaks[i]
        seg = sig_demean[s:e]
        duration = t[e] - t[s]
        amplitude = np.mean(np.abs(seg))
        if duration >= win and amplitude < amp_thr:
            events.append((t[s], t[e]))
    return events

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
        t = data[:, 0]
        z = data[:, 1]

        fs = 20  # Sampling rate
        z_filtered = bandpass_filter(z, 0.1, 0.8, fs)
        z_demean = z_filtered - np.mean(z_filtered)

        peaks = detect_breaths(z_demean, fs)
        cen_events = detect_central_apnea(t, z_demean, peaks, fs)

        # Merge nearby events
        cen_merged = []
        if cen_events:
            start, end = cen_events[0]
            for s, e in cen_events[1:]:
                if s - end < 5:
                    end = e
                else:
                    cen_merged.append((start, end))
                    start, end = s, e
            cen_merged.append((start, end))

        # Plot for verification
        plt.figure(figsize=(12, 4))
        plt.plot(t, z_demean, label='Z Accel (Filtered)', color='black')
        plt.plot(t[peaks], z_demean[peaks], 'x', label='Breaths', color='green')
        for s, e in cen_merged:
            plt.axvspan(s, e, color='blue', alpha=0.2, label='Central Apnea')
        plt.xlabel("Time (s)")
        plt.ylabel("Z (g)")
        plt.title("Central Apnea Detection")
        plt.legend()
        plt.savefig('static/apnea_plot.png')
        plt.close()

        response = {
            'central_apnea_events': [{'start': round(s, 2), 'end': round(e, 2)} for s, e in cen_merged],
            'total_events': len(cen_merged),
            'plot_path': '/static/apnea_plot.png'
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
