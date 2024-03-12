import explorepy
import pylsl
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import csv
import time
from scipy.signal import find_peaks

# Initialize ExplorePy device
#explorer = explorepy.Explore()
#explorer.connect(device_name="Explore_8447")  # Replace with your device's Bluetooth name
#explorer.set_sampling_rate(sampling_rate=250)
#explorer.set_channels(channel_mask="00001111")  # Enable the four channels (1, 2, 3, 4)
#explorer.push2lsl(duration=50)

# Arrays to store data from the four channels
buffer_channel1 = np.empty(1000)
buffer_channel2 = np.empty(1000)
buffer_channel3 = np.empty(1000)
buffer_channel4 = np.empty(1000)

idx = 0
buffer_update_freq = 50  # ms

streams = pylsl.resolve_stream('name', 'EOGData')
inlet = pylsl.StreamInlet(streams[0])

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
derivative_result1 = np.zeros(1000)
derivative_result2 = np.zeros(1000)
line1, = ax.plot(buffer_channel1, 'g-')
line2, = ax.plot(buffer_channel3, 'm-')
ax.set_xlabel('Time')
ax.set_ylabel('Derivatives (d/dt) of Filtered Results 1-2 and 3-4')

# Set the y-axis limits to -1 and 1

# Bandpass filter parameters
lowcut = 0.1  # Lower cutoff frequency in Hz
highcut = 20.0  # Upper cutoff frequency in Hz
fs = 250.0  # Sampling rate in Hz
nyquist = 0.5 * fs
low = lowcut / nyquist
high = highcut / nyquist

# Design the bandpass filter
b, a = signal.butter(2, [low, high], btype='band')

# Variables for derivative calculation
prev_filtered_result1 = 0
prev_filtered_result2 = 0
prev_time = 0

# Taille du noyau du filtre à moyenne mobile (par exemple, 5 échantillons)
kernel_size = 36

# Créez un noyau de moyenne mobile
moving_avg_kernel = np.ones(kernel_size) / kernel_size
# Définir un facteur d'amplification pour les pics de la dérivée
amplification_factor = 2.0

# Define the variables for time tracking
start_time = time.time()
run_time = 0
capture_duration = 300  # Duration to capture data (in seconds)

# Lists to store derivative data
derivative_data = []
derivative_result1_list = []
derivative_result2_list = []
initial_capture = True

        
while run_time < capture_duration:
    start_iteration_time = time.time()  # Début de l'itératio
    sample, timestamp = inlet.pull_sample()
    buffer_channel1[idx] = sample[0]
    buffer_channel2[idx] = sample[1]
    buffer_channel3[idx] = sample[4]
    buffer_channel4[idx] = sample[0]
    idx += 1

    if idx == len(buffer_channel1):
        idx = 999 - buffer_update_freq

        # Apply bandpass filter to the data
        filtered_result1 = signal.lfilter(b, a, buffer_channel1 - buffer_channel2)
        filtered_result2 = signal.lfilter(b, a, buffer_channel2 - buffer_channel3)

        # Apply the moving average filter to the filtered results
        filtered_result1 = np.convolve(filtered_result1, moving_avg_kernel, mode='valid')
        filtered_result2 = np.convolve(filtered_result2, moving_avg_kernel, mode='valid')

        # Calculate the derivatives for filtered_result1 and filtered_result2
        current_time = timestamp
        time_diff = current_time - prev_time
        
        if time_diff > 0.08:  # Calculate the derivatives every 80 ms

            derivative1 = (filtered_result1 - prev_filtered_result1) / time_diff
            derivative2 = (filtered_result2 - prev_filtered_result2) / time_diff

            # Amplifier les pics de la dérivée en multipliant par le facteur d'amplification
            derivative1 = derivative1 * amplification_factor
            derivative2 = derivative2 * amplification_factor

            prev_filtered_result1 = filtered_result1
            prev_filtered_result2 = filtered_result2
            # Mise à jour des résultats des dérivées
            derivative_result1 = np.append(derivative_result1, derivative1)
            derivative_result2 = np.append(derivative_result2, derivative2)
            derivative_result1 = derivative_result1[-1000:]
            derivative_result2 = derivative_result2[-1000:]

                        # Ajouter seulement les 50 valeurs correspondant à buffer_update_frequency
            derivative_result1_list.extend(derivative_result1[-buffer_update_freq:])
            derivative_result2_list.extend(derivative_result2[-buffer_update_freq:])
            
                
            # Mise à jour des données y pour line1 et line2
            line1.set_ydata(buffer_channel1)
            line2.set_ydata(buffer_channel3)

            # Calcul des nouvelles limites pour l'axe y basé sur buffer_channel1 et buffer_channel2
            all_y_data = np.concatenate([buffer_channel1, buffer_channel2])
            y_min, y_max = all_y_data.min(), all_y_data.max()
            margin = (y_max - y_min) * 0.05  # Marge de 5%
            ax.set_ylim(y_min - margin, y_max + margin)

            # Rafraîchissement du graphique
            fig.canvas.flush_events()
            plt.pause(0.01)
            
            buffer_channel1[:idx+1] = buffer_channel1[buffer_update_freq:]
            buffer_channel2[:idx+1] = buffer_channel2[buffer_update_freq:]
            buffer_channel3[:idx+1] = buffer_channel3[buffer_update_freq:]
            buffer_channel4[:idx+1] = buffer_channel4[buffer_update_freq:]
            # Track the run time
            run_time = time.time() - start_time

            
csv_file = "derivative_data.csv"

with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Derivative1', 'Derivative2'])
    writer.writerows(zip(derivative_result1_list, derivative_result2_list))

# Find positive peaks in derivative_result1_list
positive_peaks1, _ = find_peaks(derivative_result1_list, height=0.0015,distance=50)
# Find negative peaks in derivative_result1_list
negative_peaks1, _ = find_peaks(np.array(derivative_result1_list) * -1, height=0.0015,distance=50)
# Combine positive and negative peaks for Channel 1-2 Derivative
all_peaks1 = np.union1d(positive_peaks1, negative_peaks1)
# Find positive peaks in derivative_result2_list
positive_peaks2, _ = find_peaks(derivative_result2_list, height=0.0015,distance=50)
# Find negative peaks in derivative_result2_list
negative_peaks2, _ = find_peaks(np.array(derivative_result2_list)*-1, height=0.0015,distance=50)
# Combine positive and negative peaks for Channel 3-4 Derivative
all_peaks2 = np.union1d(positive_peaks2, negative_peaks2)



# Function to check if two peaks are within 150 indices of each other
def is_nearby(peak1, peak2):
    return abs(peak1 - peak2) < 150
# Close the live plot window
plt.close()

plt.figure()
plt.plot(derivative_result1_list, 'g-', label='Channel 3-4 Derivative')
plt.plot(all_peaks1, [derivative_result1_list[i] for i in all_peaks1], 'bo', label='All Peaks (Channel 3-4)')
# Liste pour stocker les pics à exclure
# Liste pour stocker les pics à exclure
exclude_peaks1 = set()

# Exclure les paires de pics voisins et les pics dans les 100 premiers indices
for i in range(len(all_peaks1)):
    if all_peaks1[i] < 100:
        print(f"Excluding peaks 1-Cent: {all_peaks1[i]}")
        exclude_peaks1.add(all_peaks1[i])
    else:
        for j in range(i + 1, len(all_peaks1)):
            if is_nearby(all_peaks1[i], all_peaks1[j]):
                print(f"Excluding peaks 1-Blink: {all_peaks1[i]}, {all_peaks1[j]}")
                exclude_peaks1.add(all_peaks1[i])
                exclude_peaks1.add(all_peaks1[j])

# Plot excluded peaks in blue 'x'
plt.plot(list(exclude_peaks1), [derivative_result1_list[i] for i in exclude_peaks1], 'xg', label='Excluded Peaks (Channel 1-2)')

# Sélectionner les pics qui ne sont pas dans la liste des exclusions
peaks_to_display1 = list(set(all_peaks1) - exclude_peaks1)
# Plot selected peaks in red 'o'
plt.plot(peaks_to_display1, [derivative_result1_list[i] for i in peaks_to_display1], 'ro', label='Selected Peaks (Channel 1-2)')

plt.xlabel('Time')
plt.ylabel('Derivative 1')
plt.legend()
plt.savefig('derivative1_plot_peaks.png')
plt.show()


# Plot all peaks and the selected peaks for Channel 3-4 Derivative
plt.figure()
plt.plot(derivative_result2_list, 'g-', label='Channel 3-4 Derivative')
plt.plot(all_peaks2, [derivative_result2_list[i] for i in all_peaks2], 'bo', label='All Peaks (Channel 3-4)')

# Liste pour stocker les pics à exclure
exclude_peaks2 = set()

# Exclure les paires de pics voisins et les pics dans les 100 premiers indices
for i in range(len(all_peaks2)):
    if all_peaks2[i] < 100:
        print(f"Excluding peaks 2-Cent: {all_peaks2[i]}")
        exclude_peaks2.add(all_peaks2[i])
    else:
        for j in range(i + 1, len(all_peaks2)):
            if is_nearby(all_peaks2[i], all_peaks2[j]):
                print(f"Excluding peaks 2-Blink: {all_peaks2[i]}, {all_peaks2[j]}")
                exclude_peaks2.add(all_peaks2[i])
                exclude_peaks2.add(all_peaks2[j])

# Plot excluded peaks in blue 'x'
plt.plot(list(exclude_peaks2), [derivative_result2_list[i] for i in exclude_peaks2], 'xg', label='Excluded Peaks (Channel 3-4)')

# Sélectionner les pics qui ne sont pas dans la liste des exclusions
peaks_to_display2 = list(set(all_peaks2) - exclude_peaks2)
# Plot selected peaks in red 'o'
plt.plot(peaks_to_display2, [derivative_result2_list[i] for i in peaks_to_display2], 'ro', label='Selected Peaks (Channel 3-4)')

plt.xlabel('Time')
plt.ylabel('Derivative 2')
plt.legend()
plt.savefig('derivative2_plot_peaks.png')
plt.show()


positive_peaks1_values = [derivative_result1_list[i] for i in peaks_to_display1 if derivative_result1_list[i] > 0]
negative_peaks1_values = [derivative_result1_list[i] for i in peaks_to_display1 if derivative_result1_list[i] < 0]
positive_peaks2_values = [derivative_result2_list[i] for i in peaks_to_display2 if derivative_result2_list[i] > 0]
negative_peaks2_values = [derivative_result2_list[i] for i in peaks_to_display2 if derivative_result2_list[i] < 0]

# Calculer 80% de la valeur du pic minimal pour Channel 1-2 Derivative (seuil positif)
threshold1_plus = 0.8 * np.min(positive_peaks1_values) if positive_peaks1_values else 0.0
# Calculer 80% de la valeur du pic maximal pour Channel 1-2 Derivative (seuil négatif)
threshold1_minus = 0.8 * np.max(negative_peaks1_values) if negative_peaks1_values else 0.0
# Calculer 80% de la valeur du pic minimal pour Channel 3-4 Derivative (seuil positif)
threshold2_plus = 0.8 * np.min(positive_peaks2_values) if positive_peaks2_values else 0.0
# Calculer 80% de la valeur du pic maximal pour Channel 3-4 Derivative (seuil négatif)
threshold2_minus = 0.8 * np.max(negative_peaks2_values) if negative_peaks2_values else 0.0

# Afficher les seuils
print(f"Threshold 1 (Channel 1-2): Plus = {threshold1_plus:.6f}, Minus = {threshold1_minus:.6f}")
print(f"Threshold 2 (Channel 3-4): Plus = {threshold2_plus:.6f}, Minus = {threshold2_minus:.6f}")

print("All Peaks (Channel 1-2):", all_peaks1)
print("All Peaks (Channel 3-4):", all_peaks2)