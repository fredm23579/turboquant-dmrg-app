"""Standalone benchmark comparing SVD vs TurboQuant truncation on the 1D Heisenberg chain.

Run from the backend/ directory:
    python -m benchmark

Requires: numpy, scipy
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import time
import json
from app.solver import DMRGSolver


def run_benchmark(n_sites=16, sweeps=3):
    chi_values = [4, 8, 16, 32, 64]
    results = []

    for chi in chi_values:
        print(f"  chi_max={chi} ...", flush=True)

        # SVD solver
        svd_solver = DMRGSolver(n_sites=n_sites, chi_max=chi, J=1.0)
        t_svd_start = time.perf_counter()
        svd_energies, svd_times = svd_solver.run_simulation(mode="svd", sweeps=sweeps)
        t_svd_total = time.perf_counter() - t_svd_start

        # TurboQuant solver
        tq_solver = DMRGSolver(n_sites=n_sites, chi_max=chi, J=1.0)
        t_tq_start = time.perf_counter()
        tq_energies, tq_times = tq_solver.run_simulation(mode="turboquant", sweeps=sweeps)
        t_tq_total = time.perf_counter() - t_tq_start

        results.append({
            "chi_max":          chi,
            "svd_energy_final": svd_energies[-1],
            "tq_energy_final":  tq_energies[-1],
            "svd_time_avg":     float(np.mean(svd_times)),
            "tq_time_avg":      float(np.mean(tq_times)),
            "svd_total_s":      round(t_svd_total, 4),
            "tq_total_s":       round(t_tq_total, 4),
        })

        print(f"    SVD  energy/site = {svd_energies[-1]:.6f}  avg_trunc = {np.mean(svd_times)*1e6:.1f} µs")
        print(f"    TQ   energy/site = {tq_energies[-1]:.6f}  avg_trunc = {np.mean(tq_times)*1e6:.1f} µs")

    with open("benchmark_results.json", "w") as f:
        json.dump({"n_sites": n_sites, "sweeps": sweeps, "results": results}, f, indent=2)
    print("Results saved to benchmark_results.json")
    return results


if __name__ == "__main__":
    print("=== TurboQuant 1D DMRG Benchmark ===")
    print(f"1D Heisenberg chain, n_sites=16, sweeps=3")
    run_benchmark(n_sites=16, sweeps=3)
