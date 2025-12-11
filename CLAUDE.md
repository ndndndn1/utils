# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Semiconductor SEM Image Analysis Toolkit - browser-based tools for batch processing SEM images to detect and measure microstructure features (ETCH profiles, trench bucket shapes).

**Main Tools:**
- `index.html` - Batch Image Measurement Tool (Ray/Canny/Adaptive edge detection)
- `trench-bucket-scanner.html` - Trench Bucket Scanner (ML-based shape recognition with noise filtering)

## Architecture

- **Pure client-side** - no backend, no build system, no package.json
- **Single-file HTML apps** with embedded CSS and JavaScript
- **Canvas API** for all image processing
- **CDN dependencies:** XLSX.js, JSZip, TensorFlow.js (lazy-loaded)

### State Management Pattern
Both tools use a centralized state object:
```javascript
const state = {
    images: [],      // {name, canvas, ctx, grayData, startPoint, results, ...}
    sourceIndex: null,
    // tool-specific fields...
};
```

### Event-Driven Updates
Parameter changes auto-trigger remeasurement:
```javascript
document.getElementById('paramInput').addEventListener('change', function() {
    if (state.measured) measureAll();
});
```

## Key Algorithms

### Edge Detection (index.html)
- **Ray-casting:** Threshold-based intensity detection along rays
- **Canny:** Gaussian blur → Sobel gradients → Non-max suppression → Hysteresis
- **Adaptive:** Integral image for O(1) local mean → Binary threshold

### Shape Recognition (trench-bucket-scanner.html)
- **Statistical:** Mean/std profiles from training contours
- **Template:** NCC (Normalized Cross-Correlation) matching
- **Neural:** TensorFlow.js with MobileNet transfer learning

### Noise Filtering
- **FFT:** 1D DFT low-pass filter on radius profiles
- **Consistency:** Derivative-based outlier detection with interpolation
- **RANSAC:** Polynomial fitting with inlier/outlier separation

## Common Functions

```javascript
// Image processing (shared pattern)
toGrayscale(imageData)           // ITU-R BT.601 conversion
bilinearSample(grayData, w, h, x, y)  // Sub-pixel interpolation
calculateGlobalMean(grayData)    // Mean intensity

// Coordinate transforms
// Polar (angle, radius) ↔ Cartesian (x, y) conversions throughout

// Export
exportAll()      // ZIP with JSON + XLSX + PNG images
```

## Running the Tools

Open HTML files directly in browser - no server required:
```bash
# Local development
open index.html
# or
python3 -m http.server 8000  # then visit localhost:8000
```

## Data

Sample SEM images in `/data/1` through `/data/15`:
- 180nm semiconductor trench profiles
- 640x480 grayscale JPEGs
- Used for algorithm validation

## Code Organization

```
├── index.html                    # Batch measurement tool (~1,550 lines)
├── trench-bucket-scanner.html    # ML shape scanner (~1,720 lines)
├── ETCH_profile_ray_251002.ipynb # Python analysis notebook
├── data/                         # Sample SEM images
└── plan/                         # Development planning docs
```

## Development Notes

- Both HTML files share similar base functions (grayscale, sampling, export) - currently duplicated
- Settings panels show/hide based on selected mode via classList.toggle('hidden', condition)
- Progress feedback with async/await and setTimeout(0) for UI responsiveness
- Export uses Blob + URL.createObjectURL for file downloads


## git push (reconnect)
unset GITHUB_TOKEN
gh auth login
git push