# Done by LLM - 2026-08-29

import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# Folder containing .npy files
input_folder = 'out'
output_folder = input_folder  # save images in the same folder

# Find all .npy files
npy_files = glob.glob(os.path.join(input_folder, '*.npy'))

for file_path in npy_files:
    # Load the 2D array
    arr = np.load(file_path)
    
    # Normalize to [0,1] for visualization (avoid division by zero)
    arr_min = arr.min()
    arr_max = arr.max()
    if arr_max - arr_min > 0:
        arr_norm = (arr - arr_min) / (arr_max - arr_min)
    else:
        arr_norm = np.zeros_like(arr)  # constant array → black image
    
    # Create output path with .png extension
    output_path = file_path[:-4] + '.png'  # replace .npy with .png
    # Alternatively use os.path.splitext
    # base = os.path.splitext(file_path)[0]
    # output_path = base + '.png'
    
    # Save heatmap using matplotlib (viridis colormap by default)
    plt.imsave(output_path, arr_norm, cmap='viridis')
    print(f'Saved: {output_path}')

print('Done.')