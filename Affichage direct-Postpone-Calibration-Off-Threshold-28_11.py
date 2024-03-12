# Importing necessary libraries
import explorepy  # For interacting with an Explore device
import pylsl  # For using Lab Streaming Layer (LSL), a real-time data sharing system
import numpy as np  # Library for scientific computing
import matplotlib.pyplot as plt  # For plotting
from scipy import signal  # For signal processing
import csv  # For CSV file operations
import time  # For time-related functions
from scipy.signal import find_peaks  # For finding peaks in signals

# Initialize ExplorePy device
explorer = explorepy.Explore()
explorer.connect(device_name="Explore_8447")  # Replace with your device's Bluetooth name
explorer.set_sampling_rate(sampling_rate=250)
explorer.set_channels(channel_mask="00001111")  # Enable the four channels (1, 2, 3, 4)
explorer.push2lsl(duration=50)

# Arrays to store data from the four channels
buffer_channel1 = np.empty(1000)
buffer_channel2 = np.empty(1000)
buffer_channel3 = np.empty(1000)
buffer_channel4 = np.empty(1000)

idx = 0  # Index for data storage
buffer_update_freq = 50  # Buffer update frequency in milliseconds

streams = pylsl.resolve_stream('name', 'Explore_8447_ExG')
inlet = pylsl.StreamInlet(streams[0])  # Creating a stream inlet

plt.ion()  # Enable interactive mode for plotting
fig = plt.figure()
ax = fig.add_subplot(111)  # Creating a subplot
derivative_result1 = np.zeros(1000)  # Array for storing derivative results of channels 1-2
derivative_result2 = np.zeros(1000)  # Array for storing derivative results of channels 3-4
line1, = ax.plot(derivative_result1, 'g-')  # Line object for derivative results of channels 1-2
line2, = ax.plot(derivative_result2, 'm-')  # Line object for derivative results of channels 3-4
ax.set_xlabel('Time')  # Set x-axis label
ax.set_ylabel('Derivatives (d/dt) of Filtered Results 1-2 and 3-4')  # Set y-axis label

# Set the y-axis limits to -0.2 and 0.2
ax.set_ylim(-0.2, 0.2)

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

# Moving average filter kernel size (e.g., 5 samples)
kernel_size = 36

# Create a moving average kernel
moving_avg_kernel = np.ones(kernel_size) / kernel_size
# Define an amplification factor for the derivative peaks
amplification_factor = 2.0

# Define the variables for time tracking
start_time = time.time()
run_time = 0
capture_duration = 20  # Duration to capture data (in seconds)

# Lists to store derivative data
derivative_data = []
derivative_result1_list = []
derivative_result2_list = []
initial_capture = True

# Data capture loop
while run_time < capture_duration:
    start_iteration_time = time.time()  # Start of iteration
    sample, timestamp = inlet.pull_sample()
    buffer_channel1[idx] = sample[0]
    buffer_channel2[idx] = sample[1]
    buffer_channel3[idx] = sample[2]
    buffer_channel4[idx] = sample[3]
    idx += 1

    if idx == len(buffer_channel1):
        idx = 999 - buffer_update_freq

        # Apply bandpass filter to the data
        filtered_result1 = signal.lfilter(b, a, buffer_channel1 - buffer_channel2)
        filtered_result2 = signal.lfilter(b, a, buffer_channel3 - buffer_channel4)

        # Apply the moving average filter to the filtered results
        filtered_result1 = np.convolve(filtered_result1, moving_avg_kernel, mode='valid')
        filtered_result2 = np.convolve(filtered_result2, moving_avg_kernel, mode='valid')

        # Calculate the derivatives for filtered_result1 and filtered_result2
        current_time = timestamp
        time_diff = current_time - prev_time
        
        if time_diff > 0.08:  # Calculate the derivatives every 80 ms
            # Calculate the derivatives by dividing the difference in filtered results by the time difference
            derivative1 = (filtered_result1 - prev_filtered_result1) / time_diff
            derivative2 = (filtered_result2 - prev_filtered_result2) / time_diff

            # Amplify the derivative peaks by multiplying by the amplification factor
            derivative1 *= amplification_factor
            derivative2 *= amplification_factor

            prev_filtered_result1 = filtered_result1
            prev_filtered_result2 = filtered_result2
            # Update the derivative results arrays
            derivative_result1 = np.append(derivative_result1, derivative1)
            derivative_result2 = np.append(derivative_result2, derivative2)
            # Keep only the latest 1000 values
            derivative_result1 = derivative_result1[-1000:]
            derivative_result2 = derivative_result2[-1000:]

            # Add only the 50 values corresponding to buffer_update_freq to the list
            derivative_result1_list.extend(derivative_result1[-buffer_update_freq:])
            derivative_result2_list.extend(derivative_result2[-buffer_update_freq:])
            
            # Update the plot data
            line1.set_ydata(derivative_result1)
            line2.set_ydata(derivative_result2)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.flush_events()
            plt.pause(0.01)  # Adjust as needed for plot refresh rate
            
            # Shift the buffer to manage new incoming data
            buffer_channel1[:idx+1] = buffer_channel1[buffer_update_freq:]
            buffer_channel2[:idx+1] = buffer_channel2[buffer_update_freq:]
            buffer_channel3[:idx+1] = buffer_channel3[buffer_update_freq:]
            buffer_channel4[:idx+1] = buffer_channel4[buffer_update_freq:]
            # Track the run time
            run_time = time.time() - start_time

# Save the derivative data to a CSV file
csv_file = "derivative_data.csv"
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Derivative1', 'Derivative2'])
    writer.writerows(zip(derivative_result1_list, derivative_result2_list))

# Find peaks in the derivative data
# Find positive peaks in derivative_result1_list
positive_peaks1, _ = find_peaks(derivative_result1_list, height=0.0015, distance=50)
# Find negative peaks in derivative_result1_list
negative_peaks1, _ = find_peaks(-np.array(derivative_result1_list), height=0.0015, distance=50)
# Combine positive and negative peaks for Channel 1-2 Derivative
all_peaks1 = np.union1d(positive_peaks1, negative_peaks1)

# Repeat the process for Channel 3-4 Derivative
positive_peaks2, _ = find_peaks(derivative_result2_list, height=0.0015, distance=50)
negative_peaks2, _ = find_peaks(-np.array(derivative_result2_list), height=0.0015, distance=50)
all_peaks2 = np.union1d(positive_peaks2, negative_peaks2)

# Function to check if two peaks are within 150 indices of each other
def is_nearby(peak1, peak2):
    return abs(peak1 - peak2) < 150

# Close the live plot window
plt.close()

# Plotting and analyzing the peaks
# Plot derivative data and mark all peaks for Channel 1-2
plt.figure()
plt.plot(derivative_result1_list, 'g-', label='Channel 1-2 Derivative')
plt.plot(all_peaks1, [derivative_result1_list[i] for i in all_peaks1], 'bo', label='All Peaks (Channel 1-2)')

# Initialize a list to store peaks that will be excluded
exclude_peaks1 = set()

# Exclude nearby peak pairs and peaks within the first 100 indices
for i in range(len(all_peaks1)):
    if all_peaks1[i] < 100:
        print(f"Excluding peak near start: {all_peaks1[i]}")
        exclude_peaks1.add(all_peaks1[i])
    else:
        for j in range(i + 1, len(all_peaks1)):
            if is_nearby(all_peaks1[i], all_peaks1[j]):
                print(f"Excluding nearby peaks: {all_peaks1[i]}, {all_peaks1[j]}")
                exclude_peaks1.add(all_peaks1[i])
                exclude_peaks1.add(all_peaks1[j])

# Plot excluded peaks with a different marker
plt.plot(list(exclude_peaks1), [derivative_result1_list[i] for i in exclude_peaks1], 'xg', label='Excluded Peaks (Channel 1-2)')

# Select peaks not in the exclusion list
peaks_to_display1 = list(set(all_peaks1) - exclude_peaks1)
# Plot selected peaks with a distinct marker
plt.plot(peaks_to_display1, [derivative_result1_list[i] for i in peaks_to_display1], 'ro', label='Selected Peaks (Channel 1-2)')

plt.xlabel('Time')
plt.ylabel('Derivative 1')
plt.legend()
plt.savefig('derivative1_plot_peaks.png')
plt.show()

# Repeat the peak analysis for Channel 3-4 Derivative
plt.figure()
plt.plot(derivative_result2_list, 'g-', label='Channel 3-4 Derivative')
plt.plot(all_peaks2, [derivative_result2_list[i] for i in all_peaks2], 'bo', label='All Peaks (Channel 3-4)')
exclude_peaks2 = set()

# Exclude process for Channel 3-4
for i in range(len(all_peaks2)):
    if all_peaks2[i] < 100:
        print(f"Excluding peak near start: {all_peaks2[i]}")
        exclude_peaks2.add(all_peaks2[i])
    else:
        for j in range(i + 1, len(all_peaks2)):
            if is_nearby(all_peaks2[i], all_peaks2[j]):
                print(f"Excluding nearby peaks: {all_peaks2[i]}, {all_peaks2[j]}")
                exclude_peaks2.add(all_peaks2[i])
                exclude_peaks2.add(all_peaks2[j])

plt.plot(list(exclude_peaks2), [derivative_result2_list[i] for i in exclude_peaks2], 'xg', label='Excluded Peaks (Channel 3-4)')
peaks_to_display2 = list(set(all_peaks2) - exclude_peaks2)
plt.plot(peaks_to_display2, [derivative_result2_list[i] for i in peaks_to_display2], 'ro', label='Selected Peaks (Channel 3-4)')

plt.xlabel('Time')
plt.ylabel('Derivative 2')
plt.legend()
plt.savefig('derivative2_plot_peaks.png')
plt.show()

# Calculate thresholds based on the peak values
# Calculate 80% of the minimum positive peak value for Channel 1-2 Derivative (positive threshold)
threshold1_plus = 0.8 * np.min([v for v in positive_peaks1_values if v > 0], default=0.0)
# Calculate 80% of the maximum negative peak value for Channel 1-2 Derivative (negative threshold)
threshold1_minus = 0.8 * np.max([v for v in negative_peaks1_values if v < 0], default=0.0)
# Repeat threshold calculations for Channel 3-4 Derivative
threshold2_plus = 0.8 * np.min([v for v in positive_peaks2_values if v > 0], default=0.0)
threshold2_minus = 0.8 * np.max([v for v in negative_peaks2_values if v < 0], default=0.0)

# Display thresholds
print(f"Threshold 1 (Channel 1-2): Plus = {threshold1_plus:.6f}, Minus = {threshold1_minus:.6f}")
print(f"Threshold 2 (Channel 3-4): Plus = {threshold2_plus:.6f}, Minus = {threshold2_minus:.6f}")

print("All Peaks (Channel 1-2):", all_peaks1)
print("All Peaks (Channel 3-4):", all_peaks2)
