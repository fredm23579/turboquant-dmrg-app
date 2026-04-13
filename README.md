# 🌌 TurboQuant-DMRG — Quantum Speedup Engine

**TurboQuant-DMRG** is a high-performance simulation platform that applies the **TurboQuant** vector quantization algorithm to the **Density Matrix Renormalization Group (DMRG)** method. By replacing traditional $O(\chi^3)$ Singular Value Decomposition (SVD) with an $O(\chi \log \chi)$ random rotation and scalar quantization strategy, it achieves order-of-magnitude speedups in solving quantum spin Hamiltonians with minimal loss in accuracy.

[![Live Demo](https://img.shields.io/badge/Live-Demo-6c63ff?style=for-the-badge&logo=react)](https://turboquant-dmrg.app)
[![Tech Stack](https://img.shields.io/badge/Stack-FastAPI%20%2B%20React-00a?style=for-the-badge&logo=fastapi)](https://github.com/fredm23579/turboquant-dmrg-app)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## 🚀 Key Features

- **TurboQuant Integration**: Replaces the expensive tensor truncation bottleneck with a data-oblivious vector quantization approach.
- **Physics-First Simulation**: Solves the 1D Heisenberg model using Matrix Product States (MPS) with parallel Standard (SVD) and Turbo-Charged solvers.
- **Dynamic Speedup Analytics**: Real-time visualization of truncation time complexity vs. energy convergence accuracy.
- **Interactive Control Panel**: Adjust spin chain length, max bond dimension ($\chi$), and sweep parameters on-the-fly.

## 🛠️ Built With

### Frontend (Dashboard)
- **React (v19)** & **TypeScript**: Type-safe component architecture.
- **Vite**: Ultra-fast build tool for a smooth development experience.
- **Recharts**: High-performance SVG-based visualization for scientific metrics.
- **Lucide React**: Modern, consistent iconography.

### Backend (Simulation Engine)
- **FastAPI**: High-performance Python web framework for asynchronous operations.
- **NumPy & SciPy**: Powering the heavy linear algebra and tensor contractions.
- **Pydantic**: Robust data validation for simulation parameters.

---

## 📦 Project Architecture

```text
turboquant-dmrg-app/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   └── solver.py    # DMRG Physics Engine & TurboQuant Truncation
│   └── main.py          # FastAPI Endpoints & CORS Config
├── frontend/
│   ├── src/
│   │   ├── App.tsx      # Main Dashboard Logic & Visualizations
│   │   └── App.css      # Dark-mode Scientific UI
│   └── vite.config.ts
└── README.md
```

## 🏁 Getting Started

### 1. Prerequisites
- **Python 3.10+**: `python -m pip install fastapi uvicorn numpy scipy`
- **Node.js (v18+)**: `npm install`

### 2. Launch Backend
```bash
cd backend
python main.py
```
*API accessible at `http://localhost:8000`*

### 3. Launch Frontend
```bash
cd frontend
npm run dev
```
*Dashboard accessible at `http://localhost:5173`*

---

## 🔬 Technical Insight

In standard DMRG, truncation via SVD takes $O(\chi^3)$ time, where $\chi$ is the bond dimension. As $\chi$ increases to capture entanglement, the simulation slows down drastically. 

**TurboQuant-DMRG** introduces a **Data-Oblivious Vector Quantization** step:
1. **Random Rotation**: Spreads the tensor's "energy" across all coordinates using a Fast Walsh-Hadamard Transform ($O(\chi \log \chi)$).
2. **Scalar Quantization**: Compresses the bond rank by selecting coordinates with minimal bias.
3. **Reconstruction**: Efficiently recovers the compressed tensor for the next sweep.

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

*Engineered for the next generation of quantum many-body simulations.*
