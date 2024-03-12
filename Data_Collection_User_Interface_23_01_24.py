import multiprocessing
import csv
import numpy as np
from scipy import signal
import pylsl
import explorepy
import pygame
import pandas as pd
import time
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def replace_stationary(df, colonne):
    valeurs = df[colonne].values
    premiere_valeur_non_stationary = next((val for val in valeurs if val != 'Stationary'), None)
    
    for i in range(len(valeurs)):
        if valeurs[i] == 'Stationary':
            prev_val = next((val for val in valeurs[:i][::-1] if val != 'Stationary'), None)
            next_val = next((val for val in valeurs[i+1:] if val != 'Stationary'), None)
            if i == 0:  # Si 'Stationary' est au début, utilisez la première valeur non-'Stationary'
                valeurs[i] = premiere_valeur_non_stationary
            elif prev_val is not None and next_val is not None:
                valeurs[i] = prev_val
            elif prev_val is not None:
                valeurs[i] = prev_val
            elif next_val is not None:
                valeurs[i] = next_val
    df[colonne] = valeurs
    return df

def filter_direction_sequences(df, column_name):
    # Grouper par Directions Consécutives
    df['group'] = (df[column_name] != df[column_name].shift()).cumsum()
    grouped = df.groupby('group')

    # Filtrer les Groupes
    filtered_groups = [group for _, group in grouped if len(group) >= 3]

    # Reconstruire le DataFrame
    filtered_data = pd.concat(filtered_groups)

    # Supprimer la colonne temporaire 'group'
    filtered_data = filtered_data.drop(columns=['group'])

    return filtered_data

def create_interpolated_position_data(input_csv_path, output_csv_path):
    # Étape 1: Lire les données
    position_data = pd.read_csv(input_csv_path)

    # Étape 2: Filtrer les séquences de direction
    position_data = filter_direction_sequences(position_data, 'Direction')

    target_interval = 1 / 250  # secondes
    new_rows = []

    for i in range(len(position_data) - 1):
        current_row = position_data.iloc[i]
        next_row = position_data.iloc[i + 1]
        time_diff = next_row['Elapsed Time'] - current_row['Elapsed Time']
        num_points_to_interpolate = int(time_diff / target_interval) - 1

        new_rows.append(current_row.to_dict())

        for j in range(1, num_points_to_interpolate + 1):
            factor = j * target_interval / time_diff
            new_row = {}
            new_row['Elapsed Time'] = current_row['Elapsed Time'] + factor * time_diff
            new_row['x'] = current_row['x'] + factor * (next_row['x'] - current_row['x'])
            new_row['y'] = current_row['y'] + factor * (next_row['y'] - current_row['y'])
            new_row['Position'] = current_row['Position'] if factor < 0.5 else next_row['Position']
            new_row['Blink'] = current_row['Blink'] if factor < 0.5 else next_row['Blink']
            new_row['Direction'] = current_row['Direction'] if factor < 0.5 else next_row['Direction']
            new_rows.append(new_row)

    new_rows.append(position_data.iloc[-1].to_dict())
    interpolated_data = pd.DataFrame(new_rows)
    interpolated_data.to_csv(output_csv_path, index=False)

def create_file_name(base_name, duration, current_time, patient_name, electrodes_positions, electrode_type, file_type='csv'):
    # Format and sanitize file name
    patient_name = patient_name.replace(" ", "_")
    electrodes_positions = electrodes_positions.replace(" ", "").replace(",", "_")
    electrode_type = electrode_type.replace(" ", "_")
    return f"{base_name}_{duration}_{current_time}_{patient_name}_{electrodes_positions}_{electrode_type}.{file_type}"

def create_directory(duration, patient_name, electrodes_positions,electrode_type):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Replace spaces with underscores for directory name
    patient_name = patient_name.replace(" ", "_")
    electrodes_positions = electrodes_positions.replace(" ", "").replace(",", "_")
    directory_name = f"Experience_Directions_Data_{duration}_{current_time}_{patient_name}_{electrodes_positions}_{electrode_type}"
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    return directory_name

def plot_derivative_with_position(converted_path, directory, duration, patient_name, electrodes_positions, electrode_type):
    # Plotting derivatives with respect to position
    df = pd.read_csv(converted_path)
    colors = {
        'Center-Center': 'red', 
        'Center-Right': 'blue', 
        'Center-Left': 'green', 
        'Bottom-Center': 'orange', 
        'Bottom-Left': 'purple', 
        'Bottom-Right': 'brown', 
        'Top-Center': 'pink', 
        'Top-Left': 'gray', 
        'Top-Right': 'cyan',
        'Blink': 'white',
    }

    plt.figure(figsize=(12, 6))
    ymin, ymax = df[['Derivative1', 'Derivative2']].min().min(), df[['Derivative1', 'Derivative2']].max().max()

    prev_time = df['Elapsed Time'][0]
    for index, row in df.iterrows():
        current_time = row['Elapsed Time']
        color = colors.get(row['Position-Blink'], 'black')
        plt.fill_betweenx([ymin, ymax], prev_time, current_time, color=color, alpha=0.3)
        prev_time = current_time

    plt.plot(df['Elapsed Time'], df['Derivative1'], color='black', linestyle='-')
    plt.plot(df['Elapsed Time'], df['Derivative2'], color='black', linestyle='--')
    plt.xlabel('Elapsed Time')
    plt.ylabel('Derivative Values')
    plt.title('Derivative1 and Derivative2 over Time with Background Colored by Position and Blink Status')

    legend_elements = [Line2D([0], [0], color='black', linestyle='-', label='Derivative1'),
                   Line2D([0], [0], color='black', linestyle='--', label='Derivative2')] + \
                  [Patch(facecolor=color, edgecolor=color, label=position, alpha=0.7) for position, color in colors.items()]

    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')


    plt.tight_layout()
    plot_file_name = create_file_name('derivatives_plot', duration, current_time, patient_name, electrodes_positions, electrode_type, 'png')
    plt.savefig(os.path.join(directory, plot_file_name))
    
def plot_derivative_with_direction(converted_path, directory, duration, patient_name, electrodes_positions, electrode_type):
    # Read the data from the specified file path
    df = pd.read_csv(converted_path)

    # Define a color palette for each direction
    colors = {
        'Right': 'blue', 
        'Left': 'green', 
        'Bottom': 'orange', 
        'Bottom-Left': 'yellow', 
        'Bottom-Right': 'brown', 
        'Top': 'pink', 
        'Top-Left': 'purple', 
        'Top-Right': 'cyan',
        'Stationary': 'gray',
        'Blink': 'white',
    }

    # Create a plot
    plt.figure(figsize=(12, 6))

    # Determine the y-axis limits
    ymin, ymax = df[['Derivative1', 'Derivative2']].min().min(), df[['Derivative1', 'Derivative2']].max().max()

    # Color the background for each data point based on direction
    prev_time = df['Elapsed Time'][0]
    for index, row in df.iterrows():
        current_time = row['Elapsed Time']
        color = colors.get(row['Direction-Blink'], 'black')
        plt.fill_betweenx([ymin, ymax], prev_time, current_time, color=color, alpha=0.3)
        prev_time = current_time

    # Plot the derivative lines
    plt.plot(df['Elapsed Time'], df['Derivative1'], color='black', linestyle='-')  # Solid line
    plt.plot(df['Elapsed Time'], df['Derivative2'], color='black', linestyle='--')  # Dashed line

    plt.xlabel('Elapsed Time')
    plt.ylabel('Derivative Values')
    plt.title('Derivative1 and Derivative2 over Time with Background Colored by Direction Status')

    # Add a legend for directions with reduced transparency
    legend_elements = [Line2D([0], [0], color='black', linestyle='-', label='Derivative1'),
                   Line2D([0], [0], color='black', linestyle='--', label='Derivative2')] + \
                  [Patch(facecolor=color, edgecolor=color, label=direction, alpha=0.7) for direction, color in colors.items()]

    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')


    plt.tight_layout()
    # Save the plot to the specified directory
    plot_file_name = create_file_name('directions_plot', duration, current_time, patient_name, electrodes_positions, electrode_type, 'png')
    plt.savefig(os.path.join(directory, plot_file_name))

def pygame_script(start_time, duration, directory, current_time, patient_name, electrodes_positions, electrode_type):
    # Initialize Pygame
    pygame.init()
    size = width, height = 1920, 1080
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Tracking Exercise")

    # Define colors
    BLUE, RED, WHITE = (0, 0, 255), (255, 0, 0), (175, 175, 175)
    running = True
    dot_pos = [width // 2, height // 2]
    old_dot_pos = [width // 2, height // 2]
    record_interval = 1 / 250  # Record data every 4 milliseconds

    # Define target positions
    target_positions = [(0.1, 0.1), (0.5, 0.1), (0.9, 0.1), (0.1, 0.5), (0.5, 0.5), (0.9, 0.5), (0.1, 0.9), (0.5, 0.9), (0.9, 0.9)]
    current_target_index = 4  # Start from center

    # Blink request variables
    blink_request = False
    blink_duration = 1.35  # Duration of a blink
    position_count = 0  # Counter for positions reached

    # Create and open a file for recording position data
    position_file_name = create_file_name('position_data', duration, current_time, patient_name, electrodes_positions, electrode_type)
    with open(os.path.join(directory, position_file_name), 'w', newline='') as position_file:
        position_writer = csv.writer(position_file)
        position_writer.writerow(['Elapsed Time', 'Position', 'Blink', 'x', 'y', 'Direction'])
        last_record_time = 0
        clock = pygame.time.Clock()

        while running and (time.time() - start_time) < duration:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            screen.fill(WHITE)

            current_time1 = time.time()
            elapsed_time = current_time1 - start_time

            # Record data at specified intervals
            if elapsed_time - last_record_time >= record_interval:
                horizontal = 'Left' if dot_pos[0] < width / 3 else 'Right' if dot_pos[0] > 2 * width / 3 else 'Center'
                vertical = 'Top' if dot_pos[1] < height / 3 else 'Bottom' if dot_pos[1] > 2 * height / 3 else 'Center'
                position_name = f'{vertical}-{horizontal}'.strip('-')

                dx = dot_pos[0] - old_dot_pos[0]
                dy = dot_pos[1] - old_dot_pos[1]

                # Calculate the direction
                direction = ''
                if abs(dx) > 0.96:  # Threshold for horizontal movements
                    direction += 'Right' if dx > 0 else 'Left'
                if abs(dy) > 0.54:  # Threshold for vertical movements
                    direction = ('Bottom' if dy > 0 else 'Top') + ('-' + direction if direction else '')

                if direction == '':
                    direction = 'Stationary'

                position_writer.writerow([elapsed_time, position_name, 'Blink' if blink_request else 'No-Blink', dot_pos[0], dot_pos[1], direction])
                last_record_time = elapsed_time
                old_dot_pos = list(dot_pos)

            # Check if the dot has reached the target position
            if abs(dot_pos[0] - target_positions[current_target_index][0] * width) < 5 and abs(dot_pos[1] - target_positions[current_target_index][1] * height) < 5:
                position_count += 1
                if position_count % 6 == 0:
                    blink_request = True
                    last_blink_time = current_time1

                # Choose a new random nearby target position
                nearby_positions = [i for i in range(len(target_positions)) if abs(target_positions[i][0] - target_positions[current_target_index][0]) <= 0.4 and abs(target_positions[i][1] - target_positions[current_target_index][1]) <= 0.4]
                current_target_index = np.random.choice(nearby_positions)

            
            # Move the dot towards the target
            if not blink_request:
                # Draw target circles
                for pos in target_positions:
                    pygame.draw.circle(screen, RED, [int(pos[0] * width), int(pos[1] * height)], 10)

                target_pos = [target_positions[current_target_index][0] * width, target_positions[current_target_index][1] * height]
                dot_pos[0] += (target_pos[0] - dot_pos[0]) / 20
                dot_pos[1] += (target_pos[1] - dot_pos[1]) / 20
                pygame.draw.circle(screen, BLUE, [int(dot_pos[0]), int(dot_pos[1])], 20)
            else:
                if current_time1 - last_blink_time < blink_duration:
                    # Display 'Blink' message
                    font = pygame.font.Font(None, 36)
                    text = font.render('Blink', True, RED)
                    text_rect = text.get_rect(center=(int(dot_pos[0]), int(dot_pos[1])))
                    screen.blit(text, text_rect)
                else:
                    blink_request = False

            pygame.display.flip()
            clock.tick(100)

    pygame.quit()

def explorepy_script(start_time, duration, directory, current_time, patient_name, electrodes_positions,electrode_type):
    # Initialize ExplorePy device
    explorer = explorepy.Explore()
    explorer.connect(device_name="Explore_8447")
    explorer.set_sampling_rate(sampling_rate=250)
    explorer.set_channels(channel_mask="00011111")
    explorer.push2lsl(duration)
    
        # Initialize buffers for each channel
    buffer_channel1 = np.empty(1000)
    buffer_channel2 = np.empty(1000)
    buffer_channel3 = np.empty(1000)
    buffer_channel4 = np.empty(1000)
    #buffer_timestamp = np.empty(1000)
    derivative_result1 = np.zeros(1000)
    derivative_result2 = np.zeros(1000)
    sub_result1 = np.zeros(1000)
    sub_result2 = np.zeros(1000)
    buffer_idx = 0
    buffer_update_freq = 50  # Frequency to update buffer
    
    # Bandpass filter parameters
    lowcut = 0.1
    highcut = 20.0
    fs = 250.0  # Sampling rate
    nyquist = 0.5 * fs
    b, a = signal.butter(2, [lowcut / nyquist, highcut / nyquist], btype='band')

    # Moving average filter parameters
    kernel_size = 36
    moving_avg_kernel = np.ones(kernel_size) / kernel_size
    amplification_factor = 2.0

    # LSL stream setup
    streams = pylsl.resolve_stream('name', 'Explore_8447_ExG')
    inlet = pylsl.StreamInlet(streams[0])

    # Time tracking and storage lists
    derivative_result1_list = []
    derivative_result2_list = []
    sub_result1_list = []
    sub_result2_list = []

    prev_filtered_result1 = 0  # Adjust size based on expected output after convolution
    prev_filtered_result2 = 0
    prev_time = 0

    end_time = start_time + duration
    first=None
    
    while time.time() < end_time:
        sample, timestamp = inlet.pull_sample()
        if first == None :
            start_elapsed=time.time()
            first=1
        buffer_channel1[buffer_idx] = sample[0]
        buffer_channel2[buffer_idx] = sample[1]
        buffer_channel3[buffer_idx] = sample[2]
        buffer_channel4[buffer_idx] = sample[4]
        
        buffer_idx += 1
 
        if buffer_idx == len(buffer_channel1):
            #start_elapsed = time.time()
            buffer_idx = 999 - buffer_update_freq

            # Apply bandpass filter to the data
            filtered_result1 = signal.lfilter(b, a, buffer_channel1 - buffer_channel2)
            filtered_result2 = signal.lfilter(b, a, buffer_channel3 - buffer_channel4)

            # Convolution operation
            filtered_result1 = np.convolve(filtered_result1, moving_avg_kernel, mode='valid')
            filtered_result2 = np.convolve(filtered_result2, moving_avg_kernel, mode='valid')

            # Calculate derivatives
            current_time1=timestamp
            time_diff = current_time1
            
            derivative1 = (filtered_result1 - prev_filtered_result1) / time_diff
            derivative2 = (filtered_result2 - prev_filtered_result2) / time_diff
            
            derivative1 = derivative1 * amplification_factor
            derivative2 = derivative2 * amplification_factor
            # Update previous values
            prev_filtered_result1 = filtered_result1
            prev_filtered_result2 = filtered_result2
                        
            # Calculer le temps écoulé depuis le début de la session pour cette valeur
            derivative_result1=np.append(derivative_result1_list, derivative1)
            derivative_result2=np.append(derivative_result2_list, derivative2)

            sub_result1=np.append(sub_result1_list, filtered_result1)
            sub_result2=np.append(sub_result2_list, filtered_result2)
            
            derivative_result1 = derivative_result1[-1000:]
            derivative_result2 = derivative_result2[-1000:]
            
            sub_result1 = sub_result1[-1000:]
            sub_result2 = sub_result2[-1000:]
            
            derivative_result1_list.extend(derivative_result1[-buffer_update_freq:])
            derivative_result2_list.extend(derivative_result2[-buffer_update_freq:])
            
            sub_result1_list.extend(sub_result1[-buffer_update_freq:])
            sub_result2_list.extend(sub_result2[-buffer_update_freq:])
            
            
            # Shift the buffer for the next iteration
            buffer_channel1[:buffer_idx + 1] = buffer_channel1[buffer_update_freq:]
            buffer_channel2[:buffer_idx + 1] = buffer_channel2[buffer_update_freq:]
            buffer_channel3[:buffer_idx + 1] = buffer_channel3[buffer_update_freq:]
            buffer_channel4[:buffer_idx + 1] = buffer_channel4[buffer_update_freq:]
    
    duration_elapsed_list = end_time - start_elapsed
    start_elapsed_time = np.arange(0, duration_elapsed_list, 0.004)
    time_diff_elapsed = start_elapsed - start_time
    start_elapsed_time += time_diff_elapsed
    derivative_file_name = create_file_name('derivative_data', duration, current_time, patient_name, electrodes_positions,electrode_type)
    
    with open(os.path.join(directory, derivative_file_name), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Elapsed Time', 'Derivative1', 'Derivative2','Sub1','Sub2'])
        writer.writerows(zip(start_elapsed_time, derivative_result1_list, derivative_result2_list,sub_result1_list,sub_result2_list))
    
    print("CSV file created : ", derivative_file_name)

    explorer.disconnect()

def main():
    # Set the duration for the data collection process
    duration = 20

    # Collecting user input for patient information and electrode configuration
    patient_name = input("Please enter the patient's name: ")
    electrode_type = input("Please enter the type of electrodes: ")
    electrodes_positions = input("Please enter the positions of the electrodes (separated by commas): ")

    # Record the start time and current time
    start_time = time.time()
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a directory for storing data based on the provided information
    directory = create_directory(duration, patient_name, electrodes_positions, electrode_type)

    # Start the pygame_script and explorepy_script in parallel processes
    pygame_process = multiprocessing.Process(target=pygame_script, args=(start_time, duration, directory, current_time, patient_name, electrodes_positions, electrode_type))
    explorepy_process = multiprocessing.Process(target=explorepy_script, args=(start_time, duration, directory, current_time, patient_name, electrodes_positions, electrode_type))
    
    pygame_process.start()  # Start the Pygame process
    explorepy_process.start()  # Start the ExplorePy process

    # Wait for both processes to complete
    pygame_process.join()
    explorepy_process.join()
    
    # Generate filenames for storing different types of data
    position_file_name = create_file_name('position_data', duration, current_time, patient_name, electrodes_positions, electrode_type)
    interpolated_position_file_name = create_file_name('interpolated_position_data', duration, current_time, patient_name, electrodes_positions, electrode_type)
    derivative_file_name = create_file_name('derivative_data', duration, current_time, patient_name, electrodes_positions, electrode_type)
    combined_file_name = create_file_name('combined_filtered_combined_data', duration, current_time, patient_name, electrodes_positions, electrode_type)

    # Create interpolated position data from the recorded position data
    create_interpolated_position_data(os.path.join(directory, position_file_name), os.path.join(directory, interpolated_position_file_name))

    # Check if the derivative file exists before proceeding
    derivative_file_path = os.path.join(directory, derivative_file_name)
    if os.path.exists(derivative_file_path):
        try:
            # Read and combine interpolated position data with derivative data
            interpolated_position_data = pd.read_csv(os.path.join(directory, interpolated_position_file_name))
            derivative_data = pd.read_csv(os.path.join(directory, derivative_file_name))

            # Adjust the 'Elapsed Time' in derivative data
            first_elapsed_time = interpolated_position_data['Elapsed Time'].iloc[0]
            derivative_data['Elapsed Time'] += first_elapsed_time
            derivative_data['Elapsed Time'] += 0.9
            # Convert 'Elapsed Time' columns to floats
            interpolated_position_data['Elapsed Time'] = interpolated_position_data['Elapsed Time'].astype(float)
            derivative_data['Elapsed Time'] = derivative_data['Elapsed Time'].astype(float)

            # Merge the two datasets based on 'Elapsed Time'
            combined_data = pd.merge_asof(interpolated_position_data.sort_values('Elapsed Time'), derivative_data.sort_values('Elapsed Time'), on='Elapsed Time')

            # Filter out rows with missing data
            filtered_data = combined_data.dropna(subset=['Position', 'Derivative1', 'Derivative2']).copy()

            # Functions to combine position and blink, direction and blink data
            def combine_position_blink(row):
                return row['Position'] if row['Blink'] == 'No-Blink' else 'Blink'
            def combine_direction_blink(row):
                return row['Direction'] if row['Blink'] == 'No-Blink' else 'Blink'
            
            # Add new columns combining position with blink status and direction with blink status
            filtered_data['Position-Blink'] = filtered_data.apply(combine_position_blink, axis=1)
            filtered_data['Direction-Blink'] = filtered_data.apply(combine_direction_blink, axis=1)
            filtered_data=replace_stationary(filtered_data,"Direction-Blink")
            # Save the processed data to CSV files
            interpolated_position_data.to_csv(os.path.join(directory, interpolated_position_file_name), index=False)
            derivative_data.to_csv(os.path.join(directory, derivative_file_name), index=False)
            filtered_data.to_csv(os.path.join(directory, combined_file_name), index=False)

            # Read the combined data for further processing
            combined_data = pd.read_csv(os.path.join(directory, combined_file_name))
            combined_data = combined_data.iloc[150:-150]  # Remove first 150 and last 150 rows
            combined_data.to_csv(os.path.join(directory, combined_file_name), index=False)

            # Generate plots based on the combined data
            converted_path = os.path.join(directory, combined_file_name)
            plot_derivative_with_position(converted_path, directory, duration, patient_name, electrodes_positions, electrode_type)
            plot_derivative_with_direction(converted_path, directory, duration, patient_name, electrodes_positions, electrode_type)

        except FileNotFoundError as e:
            print("Error reading the file:", e)
        except Exception as e:
            print("An error occurred while processing the data:", e)
    else:
        print("File not found:", derivative_file_path)

    print("Data filtering complete and CSV files saved in the directory:", directory)

if __name__ == "__main__":
    main()
