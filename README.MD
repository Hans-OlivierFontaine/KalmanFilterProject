# Kalman Filter Implementation Project

This project contains structured Python implementations of key concepts from the [Kalman Filter Tutorial](https://www.kalmanfilter.net/default.aspx) by B. Welsh. The purpose is to provide a clean and extensible codebase for learning and experimenting with Kalman Filters and related state estimation methods.

## ✅ Implemented Modules

We are implementing the tutorial progressively. The following components have been completed:

### 1. `BasicValueTracker`
A simple value tracker implemented in `lib/AlphaBetaGammaFilter.py`. It handles:
- Adding new observed values
- Calculating current average state
- Returning historical values

Useful for basic state tracking and smoothing without prediction.

### 2. `AlphaBetaFilter`
A 1D constant-velocity alpha-beta filter to estimate position and velocity:
- Tracks the state of an object in one dimension
- Includes a prediction step and a correction step
- Useful for simple motion models with lag reduction

### 3. `AlphaBetaGammaFilter` (In Progress)
A more advanced constant-acceleration model that improves on the alpha-beta filter by estimating acceleration as well:
- Further reduces estimation lag
- Useful when the tracked system exhibits varying velocities

## 🔧 Requirements

- Python 3.8+
- `numpy` for matrix math (minimal dependencies)
- `matplotlib` (optional, for visualizations)

Install with:

```bash
pip install numpy matplotlib
```

## 📁 Project Structure

```
kalman_filter_project/
├── lib/
│   └── AlphaBetaGammaFilter.py        # Contains BasicValueTracker, AlphaBetaFilter, AlphaBetaGammaFilter
│   └── Kalman1D.py                    # Contains 1D Kalman filters (static, dynamic, with noise, witout noise)
├── README.md
```

## 📈 Goals

- Follow the Kalman Filter tutorial step-by-step
- Write clean, well-documented Python code
- Use minimal external libraries
- Provide simple, real-world-inspired test examples

## 📌 Reference

Tutorial by B. Welsh:  
👉 https://www.kalmanfilter.net/default.aspx
