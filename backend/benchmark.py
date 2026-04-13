import numpy as np
import time
import json
from app.solver import DMRGSolver

def benchmark():
    results = []
    # Bond dimensions to test
    chi_values = [4, 8, 16, 32, 64, 128, 256]
    
    for chi in chi_values:
        print(f"Benchmarking chi={chi}...")
        solver = DMRGSolver(chi_max=chi)
        # Create a sample tensor (chi, 2, chi)
        tensor = np.random.randn(chi, 2, chi)
        
        # Benchmark SVD
        start = time.perf_counter()
        for _ in range(10):
            solver.svd_truncate(tensor, chi // 2)
        t_svd = (time.perf_counter() - start) / 10.0
        
        # Benchmark TurboQuant
        start = time.perf_counter()
        for _ in range(10):
            solver.turboquant_truncate(tensor, chi // 2)
        t_tq = (time.perf_counter() - start) / 10.0
        
        results.append({
            "chi": chi,
            "t_svd": t_svd,
            "t_tq": t_tq,
            "speedup": t_svd / t_tq
        })
        
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to benchmark_results.json")

if __name__ == "__main__":
    benchmark()
