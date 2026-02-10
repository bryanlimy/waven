# Understanding the Waven Analysis Pipeline

## Overview

The waven package analyzes calcium imaging responses to **zebra noise** stimuli to rapidly characterize neuronal receptive fields in visual cortex. Here's how it works:

## Key Concepts

### 1. Zebra Noise Stimulus
- **What it is**: A dynamic visual stimulus with sharp-edged stripes (like zebra stripes)
- **Why use it**: 
  - Elicits stronger, more repeatable responses than traditional sparse noise
  - Covers more of visual parameter space efficiently
  - Allows simultaneous characterization of multiple visual features
- **Properties**: Contains multiple spatial frequencies, orientations, and positions

### 2. Gabor Wavelet Decomposition
- **What it does**: Decomposes the zebra noise into elementary visual features
- **Gabor wavelets** are sinusoidal gratings inside Gaussian envelopes
- Each wavelet is characterized by:
  - **Position** (x, y): Where in visual space
  - **Orientation** (θ): Edge angle (0-180°)
  - **Spatial frequency** (f): How fine/coarse the pattern is
  - **Size/Scale** (σ): Spatial extent (Gaussian envelope width)
  - **Phase** (φ): Sine (0°) or cosine (90°) phase

### 3. The Analysis Pipeline

```
Zebra Noise Video
        ↓
   Downsample (to reduce computation)
        ↓
   Wavelet Decomposition (filter with Gabor library)
        ↓
   Correlation Analysis (match wavelets with neural responses)
        ↓
   Receptive Field Maps & Tuning Curves
```

### 4. What You Get

For each neuron, you extract:

1. **Spatial Receptive Field**: A 2D map showing which visual locations drive the neuron
2. **Orientation Tuning**: Preferred edge orientation (0-180°)
3. **Size/Spatial Frequency Tuning**: Preferred spatial scale
4. **Position Tuning**: Precise RF center (azimuth, elevation)
5. **Temporal Properties**: Response dynamics

## Your Data Requirements

### Input Format

1. **Zebra noise video**: 
   - Format: .mp4 or similar
   - Should cover the visual field of interest
   - Typically ~10 minutes at 30 Hz = 18,000 frames

2. **Calcium responses**:
   - Shape: `(n_neurons, n_frames)`
   - Can be:
     - Raw ΔF/F
     - Deconvolved calcium events
     - Inferred spike rates
   - Should be temporally aligned with video frames

3. **Neuron positions** (optional but recommended):
   - Shape: `(n_neurons, 2)`
   - Format: [x, y] in pixels
   - Used for spatial organization analysis

### Important Notes

- **Frame alignment**: Your calcium data MUST be temporally aligned with the stimulus
  - If imaging at different rate than stimulus (e.g., 15 Hz imaging, 30 Hz stimulus), you need to resample
  - Make sure first frame of calcium corresponds to first frame of stimulus

- **Trial structure**: 
  - The code assumes 3 repeated trials for repeatability analysis
  - If you have different number of trials, you may need to modify `repetability_trial3()`
  - If no repeats, you can skip repeatability check

## Parameter Guide

### Visual Coverage Parameters

```python
visual_coverage = [-135, 45, 34, -34]
# Format: [azimuth_left, azimuth_right, elevation_top, elevation_bottom]
# Units: degrees of visual angle
```

- **Visual coverage**: Total visual field shown by your stimulus
- **Analysis coverage**: Subset you want to analyze (can be same as visual coverage)
  - Use subset to focus on fovea or specific retinotopic region
  - Reduces computation time

### Gabor Library Parameters

```python
N_thetas = 8              # More = finer orientation tuning (but slower)
Sigmas = [2,3,4,5,6,8]    # Different RF sizes in pixels
Frequencies = [0.015, 0.04, 0.07, 0.1]  # Spatial frequencies
```

**How to choose:**
- **N_thetas**: 8 is standard for mouse V1 (matches ~22.5° bins)
- **Sigmas**: Cover range from small (sharp RFs) to large (broad RFs)
  - Depends on your downsampling resolution
  - 2-8 pixels works well for 135×54 downsampled stimulus
- **Frequencies**: Lower = coarser patterns, higher = finer patterns
  - Mouse V1: typically 0.01-0.1 cycles/pixel works well

### Downsampling Parameters

```python
NX = 135  # Horizontal resolution
NY = 54   # Vertical resolution
```

**How to choose:**
- Balance between:
  - **Higher resolution**: More precise RFs, but much slower computation
  - **Lower resolution**: Faster, but lose fine spatial details
- 135×54 is a good default for mouse (~2.7°/pixel)
- For rat or ferret, might need higher resolution

## Common Issues & Solutions

### Issue 1: "My receptive fields look noisy"

**Possible causes:**
1. Low signal-to-noise in calcium data
2. Poor temporal alignment between stimulus and responses
3. Not enough data (too few frames or trials)

**Solutions:**
- Check repeatability scores (should be >0.3 for good neurons)
- Verify temporal alignment carefully
- Consider using deconvolved/spike-inferred data instead of raw ΔF/F
- Increase number of trials or recording duration

### Issue 2: "Analysis is very slow"

**Solutions:**
- Reduce downsampling resolution (NX, NY)
- Use fewer scales/frequencies in Gabor library
- Process in chunks if you have many neurons
- Use GPU acceleration if available (torch)

### Issue 3: "I get multiple peaks in position tuning"

**Solutions:**
- This is addressed in the paper with "retinotopic smoothing"
- Can use spatial information from neighboring neurons
- Or manually select the dominant peak

### Issue 4: "Receptive fields don't match expected retinotopy"

**Check:**
1. Visual coverage parameters match your actual stimulus
2. Neuron positions are in correct coordinate system
3. No spatial transformations in your acquisition (rotation, flip)
4. Screen coordinates match imaging coordinates

### Issue 5: "Not all neurons show clear RFs"

**This is normal!** Not all neurons will have:
- Reliable responses (some neurons are silent or noisy)
- Simple, linear receptive fields (complex cells, suppressive RFs)
- Visual responses (could be motor-related, etc.)

**Filter by:**
- Repeatability score (keep neurons with >0.3)
- Maximum RF response (set threshold)
- Visual responsiveness (compare to blank periods)

## Tips for Best Results

### 1. Data Quality
- Use high SNR calcium indicator (GCaMP7, jGCaMP8)
- Ensure good motion correction
- Remove neuropil contamination if possible
- Use spike deconvolution (e.g., CASCADE, Suite2p)

### 2. Stimulus Quality
- High contrast zebra noise
- Cover full visual field (both screens for mouse)
- At least 3 repeated trials for repeatability
- 10+ minutes of stimulus (18,000 frames at 30 Hz)

### 3. Analysis Strategy
1. Start with defaults, visualize results
2. Check repeatability - this tells you data quality
3. Look at top ~10% most repeatable neurons first
4. Adjust parameters based on what you see
5. Compare with sparse noise or drifting gratings if available

### 4. Visualization
- Always plot example neurons to sanity check
- Look for:
  - Single, well-defined RF peaks
  - Smooth orientation tuning
  - Consistent position across trials
  - Reasonable RF sizes for cell type

## Comparison with Other Methods

| Method | Pros | Cons |
|--------|------|------|
| **Sparse Noise** | Simple, interpretable | Slow, only position/polarity |
| **Drifting Gratings** | Good for orientation | Only orientation, not position |
| **Natural Movies** | Realistic | Hard to interpret, need long recordings |
| **Zebra Noise** | Fast, comprehensive | Requires specific analysis pipeline |

**When to use Zebra Noise:**
- Need full RF characterization (position, orientation, size, frequency)
- Want to map many neurons simultaneously
- Time-limited experiments
- Studying RF organization across populations

## Code Architecture

```
waven/
├── WaveletGenerator.py
│   ├── makeFilterLibrary()      # Create Gabor wavelets
│   ├── downsample_video_binary() # Downsample stimulus
│   └── waveletDecomposition()    # Filter stimulus with wavelets
│
├── Analysis_Utils.py
│   ├── repetability_trial3()         # Check response repeatability
│   ├── PearsonCorrelationPinkNoise() # Main RF extraction
│   ├── Plot_RF()                     # Visualize 2D RF
│   └── PlotTuningCurve()            # Visualize tuning curves
│
└── LoadPinkNoise.py
    ├── correctNeuronPos()  # Convert positions to microns
    └── coarseWavelet()     # Load wavelet-decomposed stimulus
```

## Next Steps

After extracting RFs, you can:

1. **Population analysis**:
   - Retinotopic organization
   - Orientation maps
   - Size/frequency gradients

2. **Compare conditions**:
   - Before/after learning
   - Different behavioral states
   - Cortical layers

3. **Relate to behavior**:
   - Task-relevant RFs
   - Attention modulation
   - Locomotion effects

4. **Model building**:
   - Linear-nonlinear models
   - Normalization models
   - Predictive coding

## References

- Original paper: https://www.biorxiv.org/content/10.1101/2025.07.19.665666v1
- GitHub: https://github.com/skriabineSop/waven
- Zebra noise generation: https://github.com/mwshinn/zebra_noise

## Getting Help

If you're stuck:
1. Check the example.py in the waven repo
2. Read the bioRxiv paper for methodology details
3. Open an issue on the GitHub repo
4. Check that your data format matches expected inputs
