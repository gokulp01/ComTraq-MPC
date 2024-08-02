import csv

import matplotlib.pyplot as plt
import numpy as np


def generate_dense_lawnmower_pattern(width, height, step_size, waypoints_per_line):
    pattern = []

    num_vertical_lines = int(width / step_size) + 1

    for i in range(num_vertical_lines):
        x = i * step_size
        if i % 2 == 0:  # Moving up
            for y in np.linspace(0, height, waypoints_per_line):
                pattern.append((x, y, 0.0))
        else:  # Moving down
            for y in np.linspace(height, 0, waypoints_per_line):
                pattern.append((x, y, 0.0))

    return pattern

def save_to_csv(pattern, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x', 'y'])  # Header
        writer.writerows(pattern)

def plot_pattern(pattern):
    x, y, _ = zip(*pattern)

    plt.figure(figsize=(10, 10))
    plt.plot(x, y, '-o', markersize=2)
    plt.title('Dense Lawnmower Pattern')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def main(width, height, step_size, waypoints_per_line, csv_filename='dense_lawnmower_pattern.csv'):
    pattern = generate_dense_lawnmower_pattern(width, height, step_size, waypoints_per_line)
    save_to_csv(pattern, csv_filename)
    plot_pattern(pattern)

if __name__ == "__main__":
    # Use the bounding box size you provided
    width = 44.55  # meters
    height = 28.77  # meters

    # Adjust these parameters for desired density
    step_size = 6.0  # Distance between vertical lines (meters)
    waypoints_per_line = 25  # Number of waypoints per vertical line

    csv_filename = 'dense_lawnmower_pattern_200_waypoints.csv'

    main(width, height, step_size, waypoints_per_line, csv_filename)
