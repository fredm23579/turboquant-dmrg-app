from fastapi import FastAPI
from pydantic import BaseModel
from app.solver import DMRGSolver
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimulationParams(BaseModel):
    n_sites: int = 20
    chi_max: int = 32
    sweeps: int = 5

@app.post("/simulate")
async def simulate(params: SimulationParams):
    # Run SVD solver
    svd_solver = DMRGSolver(n_sites=params.n_sites, chi_max=params.chi_max)
    svd_energies, svd_times = svd_solver.run_simulation(mode="svd", sweeps=params.sweeps)
    
    # Run TurboQuant solver
    tq_solver = DMRGSolver(n_sites=params.n_sites, chi_max=params.chi_max)
    tq_energies, tq_times = tq_solver.run_simulation(mode="turboquant", sweeps=params.sweeps)
    
    return {
        "svd": {
            "energies": svd_energies,
            "times": svd_times
        },
        "turboquant": {
            "energies": tq_energies,
            "times": tq_times
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
