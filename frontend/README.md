# 🚀 TurboQuant-DMRG Frontend

This directory contains the **React + Vite** dashboard for visualizing the performance speedup achieved by applying TurboQuant vector quantization to quantum spin Hamiltonians.

## 🛠️ Built With

- **React (v19)** & **TypeScript**: Type-safe frontend component architecture.
- **Vite**: Ultra-fast build tool for a high-performance development environment.
- **Recharts**: High-performance SVG-based visualization for scientific metrics.
- **Lucide React**: Modern, consistent iconography.

## 🏁 Getting Started

### 1. Installation
Ensure you have **Node.js (v18+)** installed.
```bash
npm install
```

### 2. Development Mode
Run the dashboard in development mode.
```bash
npm run dev
```
*Accessible at `http://localhost:5173`*

## 📦 Project Structure

```text
src/
├── App.tsx      # Core Dashboard Logic & Charts
├── App.css      # Dark-mode Scientific UI Styles
├── main.tsx     # Application Entry Point
└── index.css    # Global CSS Variables
```

## 🔧 Integration
The frontend connects to the FastAPI backend at `http://localhost:8000/simulate`. Ensure the backend is running before launching a simulation.
