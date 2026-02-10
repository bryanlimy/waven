"""
QUICK START: Analyzing Your Zebra Noise Calcium Imaging Data

This script shows the minimal steps needed to analyze your data.

YOUR DATA FORMAT:
- Zebra noise video: .mp4 file
- Calcium responses: numpy array, shape (n_neurons, n_frames)

WHAT YOU'LL GET:
- Receptive field maps for each neuron
- Tuning curves (orientation, size, position, spatial frequency)
- Response repeatability metrics
"""

import numpy as np
import waven.WaveletGenerator as wg
import waven.Analysis_Utils as au
import waven.LoadPinkNoise as lpn
import os

# ============================================================================
# YOUR DATA (UPDATE THESE PATHS)
# ============================================================================

# Input files
ZEBRA_VIDEO = "/path/to/zebra_noise.mp4"
CALCIUM_RESPONSES = "/path/to/responses.npy"  # Shape: (n_neurons, n_frames)
NEURON_POSITIONS = "/path/to/positions.npy"   # Shape: (n_neurons, 2) [x, y pixels]

# Output directory
OUTPUT_DIR = "/path/to/output"

# ============================================================================
# PARAMETERS (adjust if needed)
# ============================================================================

# Visual field parameters (degrees)
VISUAL_COVERAGE = [-135, 45, 34, -34]      # [azi_left, azi_right, elev_top, elev_bottom]
ANALYSIS_COVERAGE = [-135, 0, 34, -34]     # Region to analyze (can be subset)

# Downsampling dimensions
NX = 135  # Downsampled width
NY = 54   # Downsampled height

# Microscope
RESOLUTION = 1.3671  # microns per pixel

# Stimulus timing
N_FRAMES = 18000  # Total frames at 30 Hz

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def run_analysis():
    """Complete analysis pipeline."""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*70)
    print("ZEBRA NOISE RECEPTIVE FIELD ANALYSIS")
    print("="*70)
    
    # ----------------------------------------------------------------------
    # STEP 1: Create Gabor Library (only need to do once)
    # ----------------------------------------------------------------------
    library_path = os.path.join(OUTPUT_DIR, "gabor_library.npy")
    
    if not os.path.exists(library_path):
        print("\nStep 1: Creating Gabor wavelet library...")
        print("This may take a few minutes...")
        
        # Standard parameters for mouse V1
        thetas = np.linspace(0, 180, 8, endpoint=False)  # 8 orientations
        sigmas = [2, 3, 4, 5, 6, 8]  # 6 scales
        frequencies = [0.015, 0.04, 0.07, 0.1]  # 4 spatial frequencies
        phases = [0, 90]  # 2 phases (for complex cell analysis)
        
        # Create spatial grid
        xs = np.arange(NX)
        ys = np.arange(NY)
        
        # Build library
        library_parts = []
        for f in frequencies:
            L = wg.makeFilterLibrary(xs, ys, thetas, sigmas, phases, f, freq=(f != 0))
            library_parts.append(L)
        
        library = np.concatenate(library_parts, axis=0)
        np.save(library_path, library)
        print(f"✓ Library created: {library.shape}")
    else:
        print(f"\nStep 1: Using existing library: {library_path}")
    
    # ----------------------------------------------------------------------
    # STEP 2: Preprocess Stimulus
    # ----------------------------------------------------------------------
    print("\nStep 2: Processing zebra noise video...")
    
    downsampled_path = ZEBRA_VIDEO[:-4] + '_downsampled.npy'
    
    if not os.path.exists(downsampled_path):
        print(f"Downsampling video to {NY}×{NX}...")
        
        # Calculate crop ratios
        vc = np.array(VISUAL_COVERAGE)
        ac = np.array(ANALYSIS_COVERAGE)
        ratio_x = 1 - ((vc[0]-vc[1]) - (ac[0]-ac[1])) / (vc[0]-vc[1])
        ratio_y = 1 - ((vc[2]-vc[3]) - (ac[2]-ac[3])) / (vc[2]-vc[3])
        
        # Downsample
        wg.downsample_video_binary(
            ZEBRA_VIDEO,
            VISUAL_COVERAGE,
            ANALYSIS_COVERAGE,
            shape=(NY, NX),
            chunk_size=1000,
            ratios=(ratio_x, ratio_y)
        )
        print(f"✓ Downsampled video saved")
    
    # Perform wavelet decomposition
    video_dir = os.path.dirname(ZEBRA_VIDEO)
    wavelets_exist = os.path.exists(os.path.join(video_dir, f"wavelet_phase0.npy"))
    
    if not wavelets_exist:
        print("Performing wavelet decomposition...")
        videodata = np.load(downsampled_path)
        sigmas = [2, 3, 4, 5, 6, 8]
        
        wg.waveletDecomposition(videodata, 0, sigmas, video_dir, library_path)
        wg.waveletDecomposition(videodata, 1, sigmas, video_dir, library_path)
        print("✓ Wavelet decomposition complete")
    else:
        print("✓ Using existing wavelet decomposition")
    
    # ----------------------------------------------------------------------
    # STEP 3: Load Neural Data
    # ----------------------------------------------------------------------
    print("\nStep 3: Loading neural data...")
    
    spks = np.load(CALCIUM_RESPONSES)
    neuron_pos = np.load(NEURON_POSITIONS)
    
    print(f"Neurons: {spks.shape[0]}")
    print(f"Frames: {spks.shape[1]}")
    
    # Convert positions to microns
    neuron_pos_um = lpn.correctNeuronPos(neuron_pos, RESOLUTION)
    
    # ----------------------------------------------------------------------
    # STEP 4: Check Repeatability
    # ----------------------------------------------------------------------
    print("\nStep 4: Checking response repeatability...")
    
    # This assumes you have 3 trials in your data
    # If you don't have repeated trials, skip this step
    respcorr = au.repetability_trial3(spks, neuron_pos_um, plotting=True)
    print(f"Mean repeatability: {np.mean(respcorr):.3f}")
    
    # ----------------------------------------------------------------------
    # STEP 5: Extract Receptive Fields
    # ----------------------------------------------------------------------
    print("\nStep 5: Extracting receptive fields...")
    
    # Load wavelet-filtered stimulus
    wavelets0, wavelets1, wavelet_c = lpn.coarseWavelet(
        video_dir,
        False,  # not using fine wavelets
        NX, NY,
        27,  # azimuth positions in coarse library
        11,  # elevation positions in coarse library
        8,   # number of orientations
        6    # number of scales
    )
    
    print(f"Wavelet shape: {wavelet_c.shape}")
    
    # Average responses across trials (if applicable)
    if spks.shape[1] >= N_FRAMES:
        mean_response = np.mean(spks[:, :N_FRAMES], axis=0)
    else:
        mean_response = spks
    
    # Compute receptive fields via correlation
    print("Computing correlations...")
    
    screen_ratio = 4096 / 1536  # Adjust if your screen is different
    sigmas_deg = [s * RESOLUTION for s in [2, 3, 4, 5, 6, 8]]
    
    rfs = au.PearsonCorrelationPinkNoise(
        wavelet_c.reshape(N_FRAMES, -1),
        mean_response,
        neuron_pos_um,
        27,  # azimuth positions
        11,  # elevation positions
        6,   # number of scales
        ANALYSIS_COVERAGE,
        screen_ratio,
        sigmas_deg,
        plotting=True
    )
    
    print(f"✓ Receptive fields extracted for {len(rfs[0])} neurons")
    
    # ----------------------------------------------------------------------
    # STEP 6: Visualize Example Neurons
    # ----------------------------------------------------------------------
    print("\nStep 6: Plotting example neurons...")
    
    # Find neurons with strong responses
    rf_strengths = [np.max(rf) for rf in rfs[0]]
    top_neurons = np.argsort(rf_strengths)[-5:]  # Top 5 neurons
    
    for idx in top_neurons:
        print(f"\nNeuron {idx} (max response: {rf_strengths[idx]:.3f})")
        
        # Plot receptive field
        au.Plot_RF(rfs[0][idx], 4, title=f"Neuron {idx}")
        
        # Plot tuning curves
        tuning = au.PlotTuningCurve(
            rfs,
            idx,
            ANALYSIS_COVERAGE,
            sigmas_deg,
            screen_ratio
        )
    
    # ----------------------------------------------------------------------
    # STEP 7: Save Results
    # ----------------------------------------------------------------------
    print("\nStep 7: Saving results...")
    
    results = {
        'receptive_fields': rfs[0],
        'repeatability': respcorr,
        'neuron_positions': neuron_pos_um,
        'rf_strengths': rf_strengths,
        'parameters': {
            'visual_coverage': VISUAL_COVERAGE,
            'analysis_coverage': ANALYSIS_COVERAGE,
            'n_frames': N_FRAMES,
            'resolution': RESOLUTION
        }
    }
    
    np.save(os.path.join(OUTPUT_DIR, 'rf_results.npy'), results, allow_pickle=True)
    print(f"✓ Results saved to {OUTPUT_DIR}/rf_results.npy")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    
    return results


# ============================================================================
# ALTERNATIVE: If you already have wavelet-decomposed stimulus
# ============================================================================

def analyze_with_existing_wavelets(spks, neuron_pos, wavelet_dir):
    """
    Use this if you've already run steps 1-2 and just want to analyze
    new neural data with the same stimulus.
    
    Parameters:
    -----------
    spks : np.ndarray, shape (n_neurons, n_frames)
        Calcium responses
    neuron_pos : np.ndarray, shape (n_neurons, 2)
        Neuron positions in pixels
    wavelet_dir : str
        Directory containing wavelet decomposition files
    """
    
    print("Loading wavelet decomposition...")
    wavelets0, wavelets1, wavelet_c = lpn.coarseWavelet(
        wavelet_dir, False, NX, NY, 27, 11, 8, 6
    )
    
    # Convert positions
    neuron_pos_um = lpn.correctNeuronPos(neuron_pos, RESOLUTION)
    
    # Check repeatability
    respcorr = au.repetability_trial3(spks, neuron_pos_um, plotting=True)
    
    # Extract RFs
    mean_response = np.mean(spks[:, :N_FRAMES], axis=0) if spks.shape[1] >= N_FRAMES else spks
    
    rfs = au.PearsonCorrelationPinkNoise(
        wavelet_c.reshape(N_FRAMES, -1),
        mean_response,
        neuron_pos_um,
        27, 11, 6,
        ANALYSIS_COVERAGE,
        4096/1536,
        [s * RESOLUTION for s in [2, 3, 4, 5, 6, 8]],
        plotting=True
    )
    
    return rfs, respcorr


# ============================================================================
# RUN IT
# ============================================================================

if __name__ == "__main__":
    # Update the paths at the top of this file, then run:
    results = run_analysis()
    
    # Or if you already have processed stimulus:
    # spks = np.load(CALCIUM_RESPONSES)
    # pos = np.load(NEURON_POSITIONS)
    # results = analyze_with_existing_wavelets(spks, pos, OUTPUT_DIR)
