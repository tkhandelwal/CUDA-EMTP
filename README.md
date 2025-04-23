# EMTP-CUDA-V2

A high-performance Electromagnetic Transients Program (EMTP) solver using CUDA for multi-GPU acceleration.

## Features

- Multi-GPU acceleration for power system simulations
- Waveform relaxation with boundary nodes
- Support for various network elements (lines, transformers, loads, etc.)
- Power electronics (MMC, STATCOM, BESS, Solar PV, Wind)
- Control systems (PLL, Current, Voltage, Power controllers)
- Semiconductor device models with thermal effects
- Real-time visualization

## Requirements

- CUDA-capable GPU(s) with compute capability 6.0 or higher
- CUDA Toolkit 11.0 or newer
- C++14 compatible compiler
- CMake 3.10 or newer
- SQLite3
- Python 3.7 or newer (for visualization)
  - Required packages: dash, plotly, numpy, pandas, socket, threading

## Building

```bash
mkdir build
cd build
cmake ..
make