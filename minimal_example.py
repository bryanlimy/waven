"""
MINIMAL EXAMPLE: Analyze Your Zebra Noise Data

This shows the absolute minimum code needed to go from your data to receptive fields.

ASSUMPTIONS:
- You have zebra noise video (.mp4)
- You have calcium responses: shape (n_neurons, n_frames)
- Frames are temporally aligned with video

OUTPUTS:
- Receptive field for each neuron
- Tuning curves (orientation, position, size, frequency)
"""

from pathlib import Path

import numpy as np
from read_roi import read_roi_zip
from scipy.io import loadmat

import waven.Analysis_Utils as au
import waven.LoadPinkNoise as lpn
import waven.WaveletGenerator as wg

# ============================================================================
# YOUR DATA - UPDATE THESE 3 LINES
# ============================================================================

output_dir = Path("runs/VT333_FOV1_day1")
video_file = output_dir / "video" / "zebra_noise.avi"

# load calcium responses
calcium_data = loadmat(output_dir / "responses" / "A12deltaF_fissa.mat")
# remove the first 2 channel
calcium_data = calcium_data["deltaF"][2:]
# remove the first 4 seconds of black and grey
calcium_data = calcium_data[:, (4 * 30) :]

# load neuron coordinates
rois = read_roi_zip(output_dir / "ROI" / "copyROIs.zip")
base_x, base_y = 18.70, -47.90
neuron_positions = []
for i in range(len(rois) - 2):
    roi = rois[f"ROI_{i+3:03d}"]
    neuron_positions.append([base_x + np.mean(roi["x"]), base_y + np.mean(roi["y"])])
neuron_positions = np.array(neuron_positions, dtype=np.float32)


# ============================================================================
# MINIMAL ANALYSIS - JUST 3 STEPS
# ============================================================================

# Step 1: Create Gabor library (only once)
print("Creating Gabor library...")
gabor_file = output_dir / "gabor_library.npy"

nx, ny = 135, 54
if not gabor_file.exists():
    # Create library with standard mouse V1 parameters
    xs = np.arange(nx)
    ys = np.arange(ny)
    n_theta = 8
    thetas = np.array([(i * np.pi) / n_theta for i in range(n_theta)])
    sigmas = np.array([2, 3, 4, 5, 6, 8], dtype=int)
    phases = np.array([0, 90], dtype=int)
    freqs = np.array([0.015, 0.04, 0.07, 0.1], dtype=np.float32)

    L = wg.makeFilterLibrary2(
        xs=xs,
        ys=ys,
        thetas=thetas,
        sigmas=sigmas,
        offsets=phases,
        frequencies=freqs,
    )
    np.save(gabor_file, L)

# Step 2: Process video
print("Processing zebra noise video...")
video_dir = video_file.parent

# Downsample (if not done already)
downsampled = video_dir / f"{video_file.stem}_downsampled.npy"
if not downsampled.exists():
    visual_coverage = np.array([-135, 45, 34, -34])
    analysis_coverage = np.array([-135, 0, 34, -34])
    ratio_x = 1 - (
        (visual_coverage[0] - visual_coverage[1])
        - (analysis_coverage[0] - analysis_coverage[1])
    ) / (visual_coverage[0] - visual_coverage[1])
    ratio_y = 1 - (
        (visual_coverage[2] - visual_coverage[3])
        - (analysis_coverage[2] - analysis_coverage[3])
    ) / (visual_coverage[2] - visual_coverage[3])
    wg.downsample_video_binary(
        str(video_file),
        visual_coverage,  # Visual coverage
        analysis_coverage,  # Analysis region
        shape=(ny, nx),
        chunk_size=1000,
        ratios=(ratio_x, ratio_y),
    )

# Wavelet decompose (if not done already)
if not (video_dir / "wavelet_phase0.npy").exists():
    video = np.load(downsampled)
    wg.waveletDecomposition(video, 0, [2, 3, 4, 5, 6, 8], video_dir, gabor_file)
    wg.waveletDecomposition(video, 1, [2, 3, 4, 5, 6, 8], video_dir, gabor_file)

# Step 3: Extract receptive fields
print("Extracting receptive fields...")

# Load wavelet-decomposed stimulus
_, _, wavelets = lpn.coarseWavelet(video_dir, False, 135, 54, 27, 11, 8, 6)

# Convert positions to microns (assumes 1.25 um/pixel - adjust if needed)
pos_um = lpn.correctNeuronPos(neuron_positions, 1.25)

# Get mean response (average across trials if you have repeats)
if calcium_data.shape[1] >= 9000:
    response = np.mean(calcium_data[:, :9000], axis=0)
else:
    response = calcium_data

# Compute receptive fields
rfs = au.PearsonCorrelationPinkNoise(
    wavelets.reshape(18000, -1),  # Stimulus
    response,  # Neural responses
    pos_um,  # Neuron positions
    27,
    11,
    6,  # Grid dimensions
    [-135, 0, 34, -34],  # Analysis coverage
    1920 / 1080,  # Screen ratio
    [s * 1.25 for s in [2, 3, 4, 5, 6, 8]],  # Sizes in degrees
    plotting=True,
)

# ============================================================================
# VIEW RESULTS
# ============================================================================

print(f"\nExtracted RFs for {len(rfs[0])} neurons")

# Plot top 5 neurons
strengths = [np.max(rf) for rf in rfs[0]]
top5 = np.argsort(strengths)[-5:]

for i in top5:
    print(f"\nNeuron {i} (strength: {strengths[i]:.3f})")
    au.Plot_RF(rfs[0][i], 4, title=f"Neuron {i}")
    au.PlotTuningCurve(
        rfs,
        i,
        [-135, 0, 34, -34],
        [s * 1.3671 for s in [2, 3, 4, 5, 6, 8]],
        4096 / 1536,
    )

# Save results
np.save("receptive_fields.npy", rfs[0])
print("\nDone! RFs saved to receptive_fields.npy")

# ============================================================================
# WHAT YOU GET
# ============================================================================
"""
rfs[0] = list of 2D receptive field maps, one per neuron
Each RF map shows: which visual locations drive that neuron

To access individual neuron:
    my_neuron_rf = rfs[0][neuron_index]
    
To see which visual position drives it most:
    best_location = np.unravel_index(np.argmax(my_neuron_rf), my_neuron_rf.shape)
    
To extract tuning properties, use:
    tuning = au.PlotTuningCurve(rfs, neuron_index, ...)
"""
