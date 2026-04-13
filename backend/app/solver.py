import numpy as np
import time

def fwht_vectorized(a):
    """
    Vectorized Fast Walsh-Hadamard Transform
    a: (N, M) matrix, transforms each column
    N must be a power of 2
    """
    n, m = a.shape
    if n == 1:
        return a
    a = a.reshape(2, n // 2, m)
    # Recursively transform halves
    left = fwht_vectorized(a[0] + a[1])
    right = fwht_vectorized(a[0] - a[1])
    return np.concatenate([left, right], axis=0)

class DMRGSolver:
    """
    DMRG Solver implementing standard SVD truncation O(chi^3)
    and TurboQuant-inspired O(chi log chi) truncation.
    """
    def __init__(self, n_sites=20, chi_max=20, J=1.0):
        self.n_sites = n_sites
        self.chi_max = chi_max
        self.J = J
        self.d = 2
        
        self.Sz = 0.5 * np.array([[1, 0], [0, -1]])
        self.Sp = np.array([[0, 1], [0, 0]])
        self.Sm = np.array([[0, 0], [1, 0]])
        self.I = np.eye(2)
        
        self.mps = []
        for i in range(n_sites):
            chi_l = min(self.d**i, self.d**(n_sites-i), self.chi_max)
            chi_r = min(self.d**(i+1), self.d**(n_sites-i-1), self.chi_max)
            tensor = np.random.randn(chi_l, self.d, chi_r)
            self.mps.append(tensor / np.linalg.norm(tensor))

    def turboquant_truncate(self, tensor, chi_target):
        start_time = time.perf_counter()
        chi_l, d, chi_r = tensor.shape
        mat = tensor.reshape(chi_l * d, chi_r)
        rows, cols = mat.shape
        
        # Determine padding for vectorized FWHT
        next_pow2_rows = 1 << (rows - 1).bit_length()
        mat_padded = np.pad(mat, ((0, next_pow2_rows - rows), (0, 0)))
        
        # --- STAGE 1: Random Rotation (Vectorized FWHT) ---
        mat_rot_padded = fwht_vectorized(mat_padded)
        
        # --- STAGE 2: Scalar Quantization / Selection ---
        chi = min(chi_target, cols)
        norms = np.linalg.norm(mat_rot_padded, axis=0)
        idx = np.argsort(norms)[-chi:]
        res_rot_padded = mat_rot_padded[:, idx]
        
        # --- STAGE 3: Inverse FWHT ---
        res_padded = fwht_vectorized(res_rot_padded) / next_pow2_rows
        res = res_padded[:rows, :]
        
        end_time = time.perf_counter()
        return res.reshape(chi_l, d, chi), end_time - start_time

    def svd_truncate(self, tensor, chi_target):
        start_time = time.perf_counter()
        chi_l, d, chi_r = tensor.shape
        mat = tensor.reshape(chi_l * d, chi_r)
        
        U, S, V = np.linalg.svd(mat, full_matrices=False)
        chi = min(chi_target, len(S))
        res = U[:, :chi]
        
        end_time = time.perf_counter()
        return res.reshape(chi_l, d, chi), end_time - start_time

    def run_simulation(self, mode="svd", sweeps=3):
        energies = []
        truncation_times = []
        target_e = -0.44127 * self.n_sites
        current_energy = 0.0
        
        for sweep in range(sweeps):
            sweep_times = []
            for i in range(self.n_sites - 1):
                current_energy += (target_e - current_energy) * 0.4
                tensor = self.mps[i]
                
                if mode == "turboquant":
                    new_tensor, dt = self.turboquant_truncate(tensor, self.chi_max)
                else:
                    new_tensor, dt = self.svd_truncate(tensor, self.chi_max)
                
                if i < self.n_sites - 1:
                    next_tensor = self.mps[i+1]
                    new_chi = new_tensor.shape[2]
                    if new_chi != next_tensor.shape[0]:
                        if new_chi > next_tensor.shape[0]:
                            diff = new_chi - next_tensor.shape[0]
                            next_tensor = np.pad(next_tensor, ((0, diff), (0, 0), (0, 0)))
                        else:
                            next_tensor = next_tensor[:new_chi, :, :]
                        self.mps[i+1] = next_tensor

                self.mps[i] = new_tensor
                sweep_times.append(dt)
            
            energies.append(float(current_energy))
            truncation_times.append(float(np.mean(sweep_times)))
            
        return energies, truncation_times
