"""
Guide to Analyzing Calcium Responses to Zebra Noise Using the Waven Package

This script demonstrates how to use the waven package (https://github.com/skriabineSop/waven)
to analyze calcium imaging responses from mouse V1 neurons to zebra noise stimuli.

Key Concept:
Zebra noise is a dynamic visual stimulus with sharp-edged stripes that elicits strong,
repeatable responses in visual cortex neurons. The waven package uses Gabor wavelets to 
decompose the stimulus and correlates these decompositions with neural responses to 
extract receptive field properties.

Required Data:
1. Zebra noise video (.mp4 or video frames)
2. Calcium responses: shape (n_neurons, n_frames)
3. Neuron positions (optional but recommended)

Installation:
-----------
conda env create -f environment.yml  # From the waven repo
python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps waven

OR manually install dependencies:
pip install scipy scikit-learn scikit-image tifffile pandas torch numpy
"""

import numpy as np
import os

# Import waven modules
import waven.WaveletGenerator as wg
import waven.Analysis_Utils as au
import waven.LoadPinkNoise as lpn

# ============================================================================
# STEP 1: Set up parameters
# ============================================================================

# Parameters for Gabor wavelet library
# These define the filters used to decompose the visual stimulus
gabor_params = {
    "N_thetas": 8,          # Number of orientations (0-180 degrees)
    "Sigmas": [2, 3, 4, 5, 6, 8],  # Gabor sizes (Gaussian envelope radius at half max, in pixels)
    "Frequencies": [0.015, 0.04, 0.07, 0.1],  # Spatial frequencies (cycles per pixel)
    "Phases": [0, 90],      # Phase values (0 and 90 degrees)
    "NX": 135,              # Number of horizontal positions (downsampled stimulus width)
    "NY": 54,               # Number of vertical positions (downsampled stimulus height)
}

# Analysis parameters
analysis_params = {
    # Screen and stimulus parameters
    "screen_x": 4096,       # Screen width in pixels
    "screen_y": 1536,       # Screen height in pixels
    "nx": 135,              # Downsampled stimulus width
    "ny": 54,               # Downsampled stimulus height
    
    # Visual field coverage (in degrees)
    # Format: [azimuth_left, azimuth_right, elevation_top, elevation_bottom]
    "visual_coverage": [-135, 45, 34, -34],    # Full visual coverage of stimulus
    "analysis_coverage": [-135, 0, 34, -34],   # Subset to analyze (can be same as visual_coverage)
    
    # Microscope parameters
    "resolution": 1.3671,   # Microscope resolution (microns per pixel)
    
    # Data parameters
    "n_frames": 18000,      # Total number of stimulus frames (at 30 Hz)
    "n_trials": 3,          # Number of repeated trials
    
    # File paths
    "movie_path": "/path/to/zebra_noise_stimulus.mp4",
    "library_path": "/path/to/gabors_library.npy",
    "spks_path": "/path/to/calcium_responses.npy",  # Shape: (n_neurons, n_frames)
    "neuron_pos_path": "/path/to/neuron_positions.npy",  # Shape: (n_neurons, 2) [x, y in pixels]
}


# ============================================================================
# STEP 2: Create or load Gabor wavelet library
# ============================================================================

def create_gabor_library(params, save_path):
    """
    Create a library of Gabor wavelets for stimulus decomposition.
    
    The library contains wavelets at different:
    - Positions (NX x NY grid)
    - Orientations (N_thetas evenly spaced from 0 to 180 degrees)
    - Sizes (different sigma values)
    - Spatial frequencies
    - Phases (0 and 90 degrees for complex cell analysis)
    """
    print("Creating Gabor library...")
    
    # Extract parameters
    nx = params["NX"]
    ny = params["NY"]
    n_thetas = params["N_thetas"]
    sigmas = params["Sigmas"]
    frequencies = params["Frequencies"]
    phases = params["Phases"]
    
    # Create evenly spaced orientations
    thetas = np.linspace(0, 180, n_thetas, endpoint=False)
    
    # Create spatial grid
    xs = np.arange(nx)
    ys = np.arange(ny)
    
    # Create library for each frequency
    # Note: Set freq=False for size-only (no frequency modulation)
    # Set freq=True for frequency-modulated Gabors
    for i, f in enumerate(frequencies):
        if f != 0:
            freq = True
        else:
            freq = False
        
        # Make filter library
        L = wg.makeFilterLibrary(xs, ys, thetas, sigmas, phases, f, freq=freq)
        
        # Save library
        if i == 0:
            library = L
        else:
            library = np.concatenate([library, L], axis=0)
    
    np.save(save_path, library)
    print(f"Gabor library saved to {save_path}")
    print(f"Library shape: {library.shape}")
    return save_path


# ============================================================================
# STEP 3: Downsample and decompose the stimulus
# ============================================================================

def preprocess_stimulus(movie_path, visual_coverage, analysis_coverage, 
                       shape, sigmas, library_path):
    """
    Downsample the zebra noise video and perform wavelet decomposition.
    
    This creates:
    1. Downsampled video
    2. Wavelet-filtered versions at different scales/orientations
    """
    print("Preprocessing stimulus...")
    
    nx, ny = shape
    
    # Calculate cropping ratios if analysis coverage differs from visual coverage
    if visual_coverage != analysis_coverage:
        visual_coverage = np.array(visual_coverage)
        analysis_coverage = np.array(analysis_coverage)
        
        ratio_x = 1 - ((visual_coverage[0] - visual_coverage[1]) - 
                      (analysis_coverage[0] - analysis_coverage[1])) / \
                      (visual_coverage[0] - visual_coverage[1])
        ratio_y = 1 - ((visual_coverage[2] - visual_coverage[3]) - 
                      (analysis_coverage[2] - analysis_coverage[3])) / \
                      (visual_coverage[2] - visual_coverage[3])
    else:
        ratio_x = 1
        ratio_y = 1
    
    print(f"Downsampling video to {ny}x{nx}...")
    # Downsample the video
    wg.downsample_video_binary(movie_path, visual_coverage, analysis_coverage,
                              shape=(ny, nx), chunk_size=1000, 
                              ratios=(ratio_x, ratio_y))
    
    # Load downsampled video
    path = os.path.dirname(movie_path)
    downsampled_path = movie_path[:-4] + '_downsampled.npy'
    videodata = np.load(downsampled_path)
    print(f"Video shape: {videodata.shape}")
    
    # Perform wavelet decomposition for phase 0 and phase 90
    print("Performing wavelet decomposition...")
    wg.waveletDecomposition(videodata, 0, sigmas, path, library_path)  # Phase 0
    wg.waveletDecomposition(videodata, 1, sigmas, path, library_path)  # Phase 90
    
    return path


# ============================================================================
# STEP 4: Load and prepare neural data
# ============================================================================

def load_neural_data(spks_path, neuron_pos_path, resolution):
    """
    Load calcium responses and neuron positions.
    
    Expected formats:
    - spks: (n_neurons, n_frames) - calcium activity (e.g., deconvolved spikes)
    - neuron_pos: (n_neurons, 2) - [x, y] positions in pixels
    """
    print("Loading neural data...")
    
    # Load calcium responses
    spks = np.load(spks_path)
    print(f"Calcium responses shape: {spks.shape}")
    print(f"Number of neurons: {spks.shape[0]}")
    print(f"Number of frames: {spks.shape[1]}")
    
    # Load neuron positions
    neuron_pos = np.load(neuron_pos_path)
    print(f"Neuron positions shape: {neuron_pos.shape}")
    
    # Convert positions from pixels to microns
    neuron_pos_um = lpn.correctNeuronPos(neuron_pos, resolution)
    print(f"Neuron positions converted to microns (resolution: {resolution} μm/pixel)")
    
    return spks, neuron_pos_um


# ============================================================================
# STEP 5: Analyze responses and extract receptive fields
# ============================================================================

def analyze_receptive_fields(spks, neuron_pos, wavelet_path, 
                            nx, ny, n_theta, n_scales,
                            analysis_coverage, screen_ratio, sigmas_deg,
                            plot=True):
    """
    Main analysis pipeline to extract receptive fields from calcium responses.
    
    Steps:
    1. Check response repeatability across trials
    2. Load wavelet-decomposed stimulus
    3. Compute correlations between wavelets and neural responses
    4. Extract tuning properties for each neuron
    """
    print("\n" + "="*70)
    print("ANALYZING RECEPTIVE FIELDS")
    print("="*70)
    
    # Step 1: Check repeatability
    print("\nStep 1: Checking response repeatability...")
    respcorr_zebra = au.repetability_trial3(spks, neuron_pos, plotting=plot)
    print(f"Mean repeatability: {np.mean(respcorr_zebra):.3f}")
    
    # Step 2: Load coarse wavelet decomposition
    print("\nStep 2: Loading wavelet-filtered stimulus...")
    wavelets0, wavelets1, wavelet_c = lpn.coarseWavelet(
        wavelet_path, False, nx, ny, 27, 11, n_theta, n_scales
    )
    print(f"Wavelet decomposition shape: {wavelet_c.shape}")
    
    # Step 3: Compute Pearson correlations
    print("\nStep 3: Computing correlations between stimulus and responses...")
    
    # Average across trials (assuming spks contains multiple trials)
    # If your data is already trial-averaged, skip this step
    n_frames_per_trial = 18000
    if spks.shape[1] >= n_frames_per_trial:
        mean_response = np.mean(spks[:, :n_frames_per_trial], axis=0)
    else:
        mean_response = spks
    
    # Reshape wavelets for correlation analysis
    wavelet_reshaped = wavelet_c.reshape(n_frames_per_trial, -1)
    
    # Compute receptive fields via correlation
    rfs_zebra = au.PearsonCorrelationPinkNoise(
        wavelet_reshaped,
        mean_response,
        neuron_pos,
        27,  # Number of azimuth positions in coarse library
        11,  # Number of elevation positions in coarse library
        n_scales,
        analysis_coverage,
        screen_ratio,
        sigmas_deg,
        plotting=plot
    )
    
    print(f"Receptive fields extracted for {len(rfs_zebra[0])} neurons")
    
    return rfs_zebra, respcorr_zebra


# ============================================================================
# STEP 6: Visualize results
# ============================================================================

def plot_neuron_rf(rfs_zebra, neuron_idx, analysis_coverage, sigmas_deg, screen_ratio):
    """
    Plot the receptive field for a specific neuron.
    
    Shows:
    - Spatial receptive field map
    - Orientation tuning
    - Size/frequency tuning
    - Position tuning (azimuth and elevation)
    """
    print(f"\nPlotting receptive field for neuron {neuron_idx}...")
    
    # Plot 2D receptive field
    au.Plot_RF(rfs_zebra[0][neuron_idx], 4, 
              title=f"Neuron {neuron_idx} - Max response: {np.max(rfs_zebra[0][neuron_idx]):.3f}")
    
    # Plot tuning curves
    tuning_curve = au.PlotTuningCurve(
        rfs_zebra, 
        neuron_idx, 
        analysis_coverage, 
        sigmas_deg, 
        screen_ratio
    )
    
    return tuning_curve


# ============================================================================
# EXAMPLE USAGE FOR YOUR DATA
# ============================================================================

def analyze_my_data():
    """
    Complete workflow for analyzing your calcium imaging data.
    
    Before running, make sure you have:
    1. Zebra noise video file
    2. Calcium responses (n_neurons, n_frames) - can be raw ΔF/F, deconvolved, etc.
    3. Neuron positions (n_neurons, 2) in pixels
    """
    
    # === UPDATE THESE PATHS WITH YOUR DATA ===
    movie_path = "/path/to/your/zebra_noise_video.mp4"
    spks_path = "/path/to/your/calcium_responses.npy"  # Shape: (n_neurons, n_frames)
    neuron_pos_path = "/path/to/your/neuron_positions.npy"  # Shape: (n_neurons, 2)
    
    # Output directory
    output_dir = "/path/to/output"
    os.makedirs(output_dir, exist_ok=True)
    
    library_path = os.path.join(output_dir, "gabors_library.npy")
    
    # === STEP 1: Create Gabor library (only need to do once) ===
    if not os.path.exists(library_path):
        create_gabor_library(gabor_params, library_path)
    else:
        print(f"Using existing Gabor library: {library_path}")
    
    # === STEP 2: Preprocess stimulus ===
    wavelet_path = preprocess_stimulus(
        movie_path,
        analysis_params["visual_coverage"],
        analysis_params["analysis_coverage"],
        (analysis_params["nx"], analysis_params["ny"]),
        gabor_params["Sigmas"],
        library_path
    )
    
    # === STEP 3: Load neural data ===
    spks, neuron_pos = load_neural_data(
        spks_path,
        neuron_pos_path,
        analysis_params["resolution"]
    )
    
    # === STEP 4: Analyze receptive fields ===
    # Calculate some additional parameters
    screen_ratio = analysis_params["screen_x"] / analysis_params["screen_y"]
    sigmas_deg = [s * analysis_params["resolution"] for s in gabor_params["Sigmas"]]
    
    rfs_zebra, respcorr = analyze_receptive_fields(
        spks,
        neuron_pos,
        wavelet_path,
        analysis_params["nx"],
        analysis_params["ny"],
        gabor_params["N_thetas"],
        len(gabor_params["Sigmas"]),
        analysis_params["analysis_coverage"],
        screen_ratio,
        sigmas_deg,
        plot=True
    )
    
    # === STEP 5: Visualize example neurons ===
    # Plot receptive field for the first few responsive neurons
    for neuron_idx in range(min(5, spks.shape[0])):
        try:
            tuning = plot_neuron_rf(
                rfs_zebra,
                neuron_idx,
                analysis_params["analysis_coverage"],
                sigmas_deg,
                screen_ratio
            )
        except Exception as e:
            print(f"Could not plot neuron {neuron_idx}: {e}")
    
    # === STEP 6: Save results ===
    results = {
        'receptive_fields': rfs_zebra,
        'repeatability': respcorr,
        'neuron_positions': neuron_pos,
        'analysis_params': analysis_params
    }
    
    results_path = os.path.join(output_dir, "rf_analysis_results.npy")
    np.save(results_path, results, allow_pickle=True)
    print(f"\nResults saved to {results_path}")
    
    return results


# ============================================================================
# SIMPLIFIED WORKFLOW IF YOU ALREADY HAVE PREPROCESSED DATA
# ============================================================================

def analyze_with_preprocessed_stimulus(spks, neuron_pos, wavelet_path, params):
    """
    If you've already run the stimulus preprocessing, use this function.
    
    Parameters:
    -----------
    spks : np.ndarray
        Calcium responses, shape (n_neurons, n_frames)
    neuron_pos : np.ndarray
        Neuron positions in microns, shape (n_neurons, 2)
    wavelet_path : str
        Path to directory containing wavelet-decomposed stimulus
    params : dict
        Analysis parameters dictionary
    """
    
    screen_ratio = params["screen_x"] / params["screen_y"]
    sigmas_deg = [s * params["resolution"] for s in gabor_params["Sigmas"]]
    
    rfs_zebra, respcorr = analyze_receptive_fields(
        spks,
        neuron_pos,
        wavelet_path,
        params["nx"],
        params["ny"],
        gabor_params["N_thetas"],
        len(gabor_params["Sigmas"]),
        params["analysis_coverage"],
        screen_ratio,
        sigmas_deg,
        plot=True
    )
    
    return rfs_zebra, respcorr


if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*70)
    print("This is a guide script. Update the paths in analyze_my_data()")
    print("and run that function to analyze your data.")
    print("="*70)
