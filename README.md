# TurboQuant-DMRG Speedup App

This application demonstrates the application of the **TurboQuant** vector quantization algorithm to the **Density Matrix Renormalization Group (DMRG)** method for solving quantum spin Hamiltonians.

## Features
- **Theoretical Speedup Visualization:** Compare $O(\chi^3)$ SVD truncation vs $O(\chi \log \chi)$ TurboQuant truncation.
- **Quantum Simulation:** Simulate 1D Heisenberg spin chains using Matrix Product States (MPS).
- **Full-Stack Dashboard:** Real-time visualization of energy convergence and truncation time metrics.

## Prerequisites
- Python 3.10+
- Node.js & npm

## Getting Started

### 1. Start the Backend (FastAPI)
```bash
cd backend
python main.py
```
The API will be available at `http://localhost:8000`.

### 2. Start the Frontend (React)
```bash
cd frontend
npm run dev
```
The dashboard will be available at `http://localhost:5173`.

## Technical Approach
Standard DMRG relies on **SVD** for bond dimension truncation, which is the computational bottleneck. This app replaces SVD with **TurboQuant**—a data-oblivious vector quantization method that uses random rotations and scalar quantization to compress tensors in linear-logarithmic time.
