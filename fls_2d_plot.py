import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import time

# Set to True for animation, False for single frame
ANIMATE = True
FRAME_DELAY = 0.05  # seconds between frames

# Read file and split by chunks (separated by empty lines)
chunks = []
current_chunk = []

with open("/Users/farhang/Downloads/fls_2d_terrain.csv", 'r') as file:
    for line in file:
        line = line.strip()
        if line:  # Non-empty line
            current_chunk.append(line)
        else:  # Empty line - end of chunk
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
    # Don't forget the last chunk if file doesn't end with empty line
    if current_chunk:
        chunks.append(current_chunk)

# Function to plot a single chunk
def plot_chunk(chunk_idx, chunks):
    plt.clf()  # Clear the plot

    # Convert selected chunk to DataFrame
    selected_chunk_data = []
    for line in chunks[chunk_idx]:
        # Try comma first, then space separation
        if ',' in line:
            parts = line.split(',')
        else:
            parts = line.split()
        selected_chunk_data.append([float(x.strip()) for x in parts])

    data = pd.DataFrame(selected_chunk_data, columns=['timestamp', 'x', 'z', 'tangent_angle', 'incident_angle', 'normal_angle'])

    # Extract data
    x = data['x'].values
    z = data['z'].values
    tangent = data['tangent_angle'].values
    incident = data['incident_angle'].values
    normal = data['normal_angle'].values

    # Plot
    plt.scatter(x, z, s=10, label='Points')

    # Plot normals (arrows)
    for i in range(0, len(x), 1): # Plot every point
        normal_vec = [math.cos(normal[i]), math.sin(normal[i])]
        tangent_vec = [math.cos(tangent[i]), math.sin(tangent[i])]
        incident_vec = [x[i], z[i]]

        # normal
        plt.arrow(x[i], z[i], normal_vec[0]*0.2, normal_vec[1]*0.2,
                  head_width=0.05, color='red', length_includes_head=True)
        # tangent
        plt.arrow(x[i], z[i], tangent_vec[0]*0.2, tangent_vec[1]*0.2,
                  head_width=0.05, color='green', length_includes_head=True)
        # incident
        # plt.arrow(0, 0, incident_vec[0], incident_vec[1],
        #           head_width=0.05, color='black', length_includes_head=True)

    plt.plot(0,0,'x',markersize=10)
    plt.title(f"2D Point Cloud with Terrain info - Chunk {chunk_idx+1}/{len(chunks)}")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()

if ANIMATE:
    plt.figure(figsize=(8, 8))
    plt.ion()  # Turn on interactive mode

    # Animate through all chunks
    while True:  # Infinite loop
        for chunk_idx in range(len(chunks)):
            plot_chunk(chunk_idx, chunks)
            plt.draw()
            # plt.savefig(f'/Users/farhang/Downloads/saved/frame_{chunk_idx:03d}.png', dpi=300, bbox_inches='tight')
            plt.pause(FRAME_DELAY)
else:
    # Single frame mode - choose which chunk to plot
    CHUNK_TO_PLOT = 402
    plt.figure(figsize=(6, 6))
    plot_chunk(CHUNK_TO_PLOT, chunks)
    plt.show()
