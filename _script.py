import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Root folder containing all your subfolders
input_folder = Path("out")

def normalize_array(arr):
    """
    Normalize an array to [0, 1] for visualization.
    Handles NaN/Inf values safely.
    """
    arr = np.asarray(arr, dtype=np.float32)

    finite_mask = np.isfinite(arr)

    if not np.any(finite_mask):
        # Everything is NaN/Inf
        return np.zeros_like(arr, dtype=np.float32)

    arr_min = np.min(arr[finite_mask])
    arr_max = np.max(arr[finite_mask])

    if arr_max - arr_min == 0:
        return np.zeros_like(arr, dtype=np.float32)

    # Replace NaN/Inf before normalization
    arr = np.nan_to_num(
        arr,
        nan=arr_min,
        posinf=arr_max,
        neginf=arr_min
    )

    return (arr - arr_min) / (arr_max - arr_min)


def convert_npy_to_png(file_path):
    """
    Load one .npy file, convert it to an image, save it as PNG,
    and delete the original .npy only if saving succeeded.
    """

    print(f"Processing: {file_path}")

    try:
        arr = np.load(file_path)

        print(f"  Shape: {arr.shape}")

        # ---------------------------------------------------------
        # Case 1: 2D array -> heatmap / spectrogram
        # Shape: (H, W)
        # ---------------------------------------------------------
        if arr.ndim == 2:
            arr_norm = normalize_array(arr)

            output_path = file_path.with_suffix(".png")

            # Save as heatmap
            plt.imsave(
                output_path,
                arr_norm,
                cmap="viridis"
            )

        # ---------------------------------------------------------
        # Case 2: 3D array -> image
        # Expected shape: (C, H, W)
        # For RGB, C = 3
        # ---------------------------------------------------------
        elif arr.ndim == 3:

            if arr.shape[0] == 3:
                # Convert CHW -> HWC
                image = np.transpose(arr, (1, 2, 0))

                # Normalize to [0, 1]
                image = normalize_array(image)

                output_path = file_path.with_suffix(".png")

                # Save RGB image
                plt.imsave(
                    output_path,
                    image
                )

            elif arr.shape[2] == 3:
                # Just in case some files are already HWC
                image = normalize_array(arr)

                output_path = file_path.with_suffix(".png")

                plt.imsave(
                    output_path,
                    image
                )

            else:
                print(
                    f"  SKIPPED: 3D array but no RGB dimension found: "
                    f"{arr.shape}"
                )
                return

        # ---------------------------------------------------------
        # Unsupported number of dimensions
        # ---------------------------------------------------------
        else:
            print(
                f"  SKIPPED: unsupported shape {arr.shape} "
                f"(expected HxW or CxHxW)"
            )
            return

        # ---------------------------------------------------------
        # If we reached here, PNG was created successfully.
        # Now delete the original .npy
        # ---------------------------------------------------------
        if output_path.exists():
            file_path.unlink()
            print(f"  Saved:   {output_path}")
            print(f"  Deleted: {file_path}")
        else:
            print(f"  ERROR: PNG was not created, keeping {file_path}")

    except Exception as e:
        print(f"  ERROR processing {file_path}: {e}")
        print(f"  Keeping original file.")


# =============================================================
# Find ALL .npy files recursively
# =============================================================

npy_files = list(input_folder.rglob("*.npy"))

print(f"Found {len(npy_files)} .npy files.\n")

for file_path in npy_files:
    convert_npy_to_png(file_path)

print("\nDone.")