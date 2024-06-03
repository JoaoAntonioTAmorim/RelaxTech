import os
import pyxdf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import filtfilt, iirnotch, iirfilter, sosfiltfilt, butter
from sklearn.ensemble import RandomForestClassifier
from mne.time_frequency import psd_array_multitaper
import time
import pyxdf
import serial

data_path = os.getcwd()

dataframes_original = {}

# Maping states 
trials_dict = {'neutral': 0, 'relaxed': 1, 'concentrating': 2}

# Filters parametes
notch_freq = 50 # Notch
quality_factor = 40
fs = 256  # Sampling rate in Hz
time_interval = 1.0 / fs  # Time interval between samples
highcut = 90 # Low-pass
lowcut = 4 # High-pass
order = 8

# Filters
b_notch, a_notch = iirnotch(notch_freq, quality_factor, fs)
sos = iirfilter(order, highcut, btype='lowpass', analog=False, ftype='butter', fs=256, output='sos')
b_hp, a_hp = butter(order, lowcut, btype='highpass', fs=256)

# Create RF classifier
rf_classifier = RandomForestClassifier(max_depth= None, min_samples_leaf=1, min_samples_split=2, n_estimators=100)

# Dictionary for filtered data
predicted_labels = []

# chunk size (sec)
chunk_size_seconds = 3
chunk_size_samples = fs * chunk_size_seconds
overlap_samples = chunk_size_samples // 2 # Overlapping chunks (50%)

# Range of interest
beta = (12, 35)

# Load data from .xdf
def load_data(file_path):
    streams, _ = pyxdf.load_xdf(file_path)
    eeg_data = None
    for stream in streams:
        if stream['info']['type'][0] == 'EEG':
            eeg_data = stream['time_series']
            break
    if eeg_data is None:
        raise ValueError("EEG data not found in the XDF file.")
    return pd.DataFrame(eeg_data)

# Low-pass, high-pass e notch
def apply_filters(data):
    data_filtrado = pd.DataFrame(sosfiltfilt(sos, data.values, axis=0), columns=data.columns)
    data_filtrado_lphp = pd.DataFrame(filtfilt(b_hp, a_hp, data_filtrado.values, axis=0), columns=data.columns)
    filtered_channel = pd.DataFrame()
    for channel in data_filtrado.columns:
        filtered_data = pd.DataFrame(filtfilt(b_notch, a_notch, data_filtrado_lphp[channel]))
        filtered_channel[channel] = filtered_data.values.flatten()
    return filtered_channel

def calculate_average_power(freq, magnitude, low_freq, high_freq):
    mask = (freq >= low_freq) & (freq <= high_freq)
    freq_interval = freq[mask]
    magnitude_interval = magnitude[mask]
    average_power = np.trapz(magnitude_interval, x=freq_interval)
    return average_power

def extract_features(data):
    multitaper_features = []
    beta_powers = []  # List for beta_power values
    
    for column in data.columns:
        psd_mt, freq_mt = psd_array_multitaper(data[column], fs, normalization='full', verbose=0)
        multitaper_features.append(psd_mt)
        beta_power = calculate_average_power(freq_mt, psd_mt, beta[0], beta[1])
        beta_powers.append(beta_power)
    
    multitaper_features = np.vstack(multitaper_features).T
    beta_powers = np.array(beta_powers).reshape(1, -1)  # Reshape
    return multitaper_features, beta_powers

# Function to send labels to Arduino
def control_led(arduino, y_pred_num):
    if y_pred_num == 0 or y_pred_num == 2:
        arduino.write(b'0')  # Send 0 to arduino
        time.sleep(0.1)  # Delay

    elif y_pred_num == 1:
        arduino.write(b'1')  # Send 1 to arduino
        time.sleep(0.1)  # Delay

def process_files(data_path):
    time_elapsed = 0  # Initializing time_elapsed
    for file in os.listdir(data_path):
        if file.endswith('.xdf'):
            file_path = os.path.join(data_path, file)
            estado = file.split('.')[0]

            # Load data .xdf
            data = load_data(file_path)

            # Check data length
            num_samples = len(data)
            if num_samples < chunk_size_samples:
                print("Data too short, skipping...")
                continue

            for i in range(0, num_samples - chunk_size_samples + 1, overlap_samples):
                chunk_data = data.iloc[i:i + chunk_size_samples, :]

                # Preprocessing
                filtered_data = apply_filters(chunk_data)

                # Extract features
                multitaper_features, beta_powers = extract_features(filtered_data)

                if len(multitaper_features) > 0:

                    x = np.arange(len(filtered_data))/fs
                    y1 = filtered_data.iloc[:,0]
                    y2 = filtered_data.iloc[:,1]
                    y3 = filtered_data.iloc[:,2]
                    y4 = filtered_data.iloc[:,3]

                    plt.cla()
                    plt.plot(x, y1, label='TP9')
                    plt.plot(x, y2, label='AF7')
                    plt.plot(x, y3, label='AF8')
                    plt.plot(x, y4, label='TP10')

                    plt.legend(loc='upper left')
                    plt.tight_layout()
                    plt.draw()
                    plt.pause(0.05)

                    all_data = np.array(beta_powers)
                    labels = [estado] * all_data.shape[0]
                    rf_classifier.fit(all_data, labels)

                    # Predicts
                    y_pred = rf_classifier.predict(all_data)
                    y_pred_num = trials_dict[y_pred[0]]
                    print("Predicted label:", y_pred_num)

                    # Arduino communication
                    control_led(arduino, y_pred_num)
                    time.sleep(0.3)

# Arduino communication
arduino = serial.Serial('COM4', 9600) 
# Run the function to process the files and plot in real time
process_files(data_path)
# Close communication
arduino.close()