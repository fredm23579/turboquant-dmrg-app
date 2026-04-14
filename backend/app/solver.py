import numpy as np
import time
from scipy.sparse.linalg import eigsh, LinearOperator

def fwht_vectorized(a):
    """Vectorized Fast Walsh-Hadamard Transform for columns of a."""
    n, m = a.shape
    if n == 1: return a
    next_pow2 = 1 << (n - 1).bit_length()
    if next_pow2 > n:
        a = np.pad(a, ((0, next_pow2 - n), (0, 0)))
        n = next_pow2
    a = a.reshape(2, n // 2, m)
    left = fwht_vectorized(a[0] + a[1])
    right = fwht_vectorized(a[0] - a[1])
    return np.concatenate([left, right], axis=0)

class DMRGSolver:
    def __init__(self, n_sites=20, chi_max=20, J=1.0):
        self.n_sites = n_sites
        self.chi_max = chi_max
        self.J = J
        self.d = 2
        
        # Operators
        self.Sz = 0.5 * np.array([[1, 0], [0, -1]])
        self.Sp = np.array([[0, 1], [0, 0]])
        self.Sm = np.array([[0, 0], [1, 0]])
        self.I = np.eye(2)
        
        # Heisenberg MPO
        self.mpo = self._build_heisenberg_mpo()
        
        # MPS Initialization (Random Right-Canonical)
        self.mps = [np.random.randn(1, self.d, 1) for _ in range(n_sites)]
        self._right_canonicalize()
        
        # Environments
        self.L = [None] * (n_sites + 1)
        self.R = [None] * (n_sites + 1)
        self.L[0] = np.ones((1, 1, 1))
        self.R[n_sites] = np.ones((1, 1, 1))

    def _build_heisenberg_mpo(self):
        # Bond dimension 5 MPO for 1D Heisenberg
        W = np.zeros((5, 5, 2, 2))
        W[0, 0] = self.I
        W[4, 4] = self.I
        W[1, 0] = self.Sz
        W[4, 1] = self.J * self.Sz
        W[2, 0] = self.Sp
        W[4, 3] = self.J * 0.5 * self.Sm # Sp-Sm term
        W[3, 0] = self.Sm
        W[4, 2] = self.J * 0.5 * self.Sp
        
        mpos = [W.copy() for _ in range(self.n_sites)]
        mpos[0] = W[4:5, :, :, :]
        mpos[-1] = W[:, 0:1, :, :]
        return mpos

    def _right_canonicalize(self):
        for i in range(self.n_sites - 1, 0, -1):
            cl, d, cr = self.mps[i].shape
            mat = self.mps[i].reshape(cl, d * cr)
            u, s, vh = np.linalg.svd(mat, full_matrices=False)
            self.mps[i] = vh.reshape(-1, d, cr)
            self.mps[i-1] = np.einsum('ijk,kl,l->ijl', self.mps[i-1], u, s)

    def update_env_right(self, site):
        self.R[site] = np.einsum('ijk, lmnj, pnr, rmk -> pli', self.mps[site], self.mpo[site], self.mps[site].conj(), self.R[site+1], optimize=True)

    def update_env_left(self, site):
        self.L[site+1] = np.einsum('pli, ijk, lmnj, pnr -> rmk', self.L[site], self.mps[site], self.mpo[site], self.mps[site].conj(), optimize=True)

    def solve_local_2site(self, i, j):
        # Effective Hamiltonian for sites i and j
        L, R = self.L[i], self.R[j+1]
        W1, W2 = self.mpo[i], self.mpo[j]
        shape = (self.mps[i].shape[0], self.d, self.d, self.mps[j].shape[2])
        size = np.prod(shape)

        def matvec(v):
            psi = v.reshape(shape)
            res = np.einsum('pqi, qrmj, rsnk, usl, ijkl -> pmnu', L, W1, W2, R, psi, optimize=True)
            return res.ravel()

        H_eff = LinearOperator((size, size), matvec=matvec)
        psi_init = np.einsum('ijk, klm -> ijlm', self.mps[i], self.mps[j], optimize=True).ravel()
        vals, vecs = eigsh(H_eff, k=1, which='SA', v0=psi_init, tol=1e-5)
        return vecs[:, 0].reshape(shape), vals[0]

    def turboquant_truncate(self, mat, chi_target):
        # Apply FWHT-based compression to the two-site tensor matrix
        rows, cols = mat.shape
        next_pow2 = 1 << (rows - 1).bit_length()
        mat_padded = np.pad(mat, ((0, next_pow2 - rows), (0, 0)))
        mat_rot = fwht_vectorized(mat_padded)
        chi = min(chi_target, cols)
        norms = np.linalg.norm(mat_rot, axis=0)
        idx = np.argsort(norms)[-chi:]
        res_rot = mat_rot[:, idx]
        res = fwht_vectorized(res_rot) / next_pow2
        return res[:rows, :]

    def run_simulation(self, mode="svd", sweeps=3):
        for i in range(self.n_sites - 1, 0, -1):
            self.update_env_right(i)
            
        energies = []
        truncation_times = []
        
        for sweep in range(sweeps):
            sweep_times = []
            # Left to Right
            for i in range(self.n_sites - 1):
                psi_2site, e = self.solve_local_2site(i, i+1)
                cl, d1, d2, cr = psi_2site.shape
                mat = psi_2site.reshape(cl * d1, d2 * cr)
                
                t0 = time.perf_counter()
                if mode == "turboquant":
                    U_hat = self.turboquant_truncate(mat, self.chi_max)
                    U, R_mat = np.linalg.qr(U_hat)
                    # For TurboQuant, we approximate the next site with the remainder
                    # To be first-principles, we must use the projection
                    V = U.T @ mat
                    S = np.ones(U.shape[1]) # SVD isn't computed here
                else:
                    U, S, V = np.linalg.svd(mat, full_matrices=False)
                    chi = min(self.chi_max, len(S))
                    U, S, V = U[:, :chi], S[:chi], V[:chi, :]
                
                sweep_times.append(time.perf_counter() - t0)
                self.mps[i] = U.reshape(cl, d1, -1)
                self.mps[i+1] = (np.diag(S) @ V).reshape(-1, d2, cr)
                self.update_env_left(i)
                
            # Right to Left
            for i in range(self.n_sites - 1, 0, -1):
                psi_2site, e = self.solve_local_2site(i-1, i)
                cl, d1, d2, cr = psi_2site.shape
                mat = psi_2site.reshape(cl * d1, d2 * cr)
                
                U, S, V = np.linalg.svd(mat, full_matrices=False)
                chi = min(self.chi_max, len(S))
                U, S, V = U[:, :chi], S[:chi], V[:chi, :]
                
                self.mps[i] = V.reshape(-1, d2, cr)
                self.mps[i-1] = (U @ np.diag(S)).reshape(cl, d1, -1)
                self.update_env_right(i)
                
            energies.append(float(e / self.n_sites)) # Energy per site
            truncation_times.append(float(np.mean(sweep_times)))
            
        return energies, truncation_times
