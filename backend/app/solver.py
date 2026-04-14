import numpy as np
import time
from scipy.sparse.linalg import eigsh, LinearOperator


def fwht_vectorized(a):
    """Vectorized Fast Walsh-Hadamard Transform applied to columns of a."""
    n, m = a.shape
    if n == 1:
        return a
    next_pow2 = 1 << (n - 1).bit_length()
    if next_pow2 > n:
        a = np.pad(a, ((0, next_pow2 - n), (0, 0)))
        n = next_pow2
    a = a.reshape(2, n // 2, m)
    left = fwht_vectorized(a[0] + a[1])
    right = fwht_vectorized(a[0] - a[1])
    return np.concatenate([left, right], axis=0)


class DMRGSolver:
    """
    Two-site DMRG solver for the 1D Heisenberg model:
        H = J * sum_i [ Sz_i Sz_{i+1} + 0.5*(Sp_i Sm_{i+1} + Sm_i Sp_{i+1}) ]

    Supports two truncation backends:
      - 'svd'        : standard O(chi^3) SVD (LAPACK)
      - 'turboquant' : O(chi^2 log chi) FWHT-based basis rotation

    MPS tensors are stored as A[i] with shape (chi_l, d, chi_r).
    Environment tensors L[i] / R[i] have shape (chi_mps, chi_mpo, chi_mps*)
    following the convention used in update_env_left / update_env_right.
    """

    def __init__(self, n_sites=16, chi_max=32, J=1.0):
        self.n_sites = n_sites
        self.chi_max = chi_max
        self.J = J
        self.d = 2

        # Spin-1/2 operators
        self.Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=float)
        self.Sp = np.array([[0, 1], [0, 0]], dtype=float)
        self.Sm = np.array([[0, 0], [1, 0]], dtype=float)
        self.I  = np.eye(2, dtype=float)

        # Build MPO and initialize MPS
        self.mpo = self._build_heisenberg_mpo()
        self.mps = [np.random.randn(1, self.d, 1).astype(float) for _ in range(n_sites)]
        self._right_canonicalize()

        # Environment tensors: L[i] is the left environment ending just before site i
        self.L = [None] * (n_sites + 1)
        self.R = [None] * (n_sites + 1)
        self.L[0]        = np.ones((1, 1, 1), dtype=float)
        self.R[n_sites]  = np.ones((1, 1, 1), dtype=float)

    # ------------------------------------------------------------------
    # MPO construction
    # ------------------------------------------------------------------
    def _build_heisenberg_mpo(self):
        """
        Bulk MPO tensor W[a, b, s, s'] (bond_dim=5) for the Heisenberg chain.

        Row/col convention (a = left MPO index, b = right MPO index):
          W[0,0] = I          (pass-through identity)
          W[4,4] = I          (pass-through identity)
          W[1,0] = Sz         (start Sz interaction)
          W[4,1] = J*Sz       (close Sz-Sz bond)
          W[2,0] = Sp         (start Sp-Sm interaction)
          W[4,2] = J/2*Sm     (close Sp-Sm bond) -- note index 2
          W[3,0] = Sm         (start Sm-Sp interaction)
          W[4,3] = J/2*Sp     (close Sm-Sp bond) -- note index 3

        Boundary sites:
          site 0  (left edge)  : only the LAST row of W matters → shape (1, 5, d, d)
          site -1 (right edge) : only the FIRST column matters  → shape (5, 1, d, d)
        """
        W = np.zeros((5, 5, 2, 2), dtype=float)
        W[0, 0] = self.I
        W[4, 4] = self.I
        W[1, 0] = self.Sz
        W[4, 1] = self.J * self.Sz
        W[2, 0] = self.Sp
        W[4, 2] = self.J * 0.5 * self.Sm
        W[3, 0] = self.Sm
        W[4, 3] = self.J * 0.5 * self.Sp

        mpos = [W.copy() for _ in range(self.n_sites)]
        # Left boundary: only outgoing (right) bond — take last row
        mpos[0]  = W[4:5, :, :, :]    # shape (1, 5, d, d)
        # Right boundary: only incoming (left) bond — take first column
        mpos[-1] = W[:, 0:1, :, :]    # shape (5, 1, d, d)
        return mpos

    # ------------------------------------------------------------------
    # MPS canonicalization
    # ------------------------------------------------------------------
    def _right_canonicalize(self):
        """Right-normalize the MPS from site n_sites-1 down to site 1."""
        for i in range(self.n_sites - 1, 0, -1):
            cl, d, cr = self.mps[i].shape
            mat = self.mps[i].reshape(cl, d * cr)
            u, s, vh = np.linalg.svd(mat, full_matrices=False)
            self.mps[i]   = vh.reshape(-1, d, cr)
            # Absorb u*s into the site to the left
            self.mps[i-1] = np.einsum('ijk,kl,l->ijl', self.mps[i-1], u, s)
        # Normalize the leftmost tensor
        norm = np.linalg.norm(self.mps[0])
        if norm > 0:
            self.mps[0] /= norm

    # ------------------------------------------------------------------
    # Environment updates
    # ------------------------------------------------------------------
    def update_env_right(self, site):
        """
        Build R[site] from R[site+1], mps[site], mpo[site].

        Index legend:
          mps[site]     : (i, j, k)   = (chi_l_mps, d, chi_r_mps)
          mpo[site]     : (l, m, n, j)= (chi_l_mpo, chi_r_mpo, d*, d)
          R[site+1]     : (p, l, q)   = (chi_r_mps, chi_r_mpo, chi_r_mps*)

        We contract:
          R[site][i, a, i*] = sum_{j,k,l,m} A[i,j,k] W[a,m,n,j] A*[i*,n,k] R[k,m,k*]
        which in einsum is:
          'ijk, amnj, ink, kml -> ial'
        but using the environment shape convention (chi_mps, chi_mpo, chi_mps*):
          R[site] shape: (chi_l_mps, chi_l_mpo, chi_l_mps*)
        """
        A  = self.mps[site]        # (cl, d, cr)
        W  = self.mpo[site]        # (wl, wr, d*, d)
        R  = self.R[site + 1]      # (cr, wr, cr*)
        # Contract: result R[site] has shape (cl, wl, cl*)
        self.R[site] = np.einsum(
            'ijk, lmnj, pnr, rml -> ipl',
            A, W, A.conj(), R, optimize=True
        )

    def update_env_left(self, site):
        """
        Build L[site+1] from L[site], mps[site], mpo[site].

          L[site]       : (cl, wl, cl*)  = (chi_l_mps, chi_l_mpo, chi_l_mps*)
          mps[site]     : (cl, d, cr)
          mpo[site]     : (wl, wr, d*, d)
          result L[site+1] shape: (cr, wr, cr*)
        """
        A  = self.mps[site]        # (cl, d, cr)
        W  = self.mpo[site]        # (wl, wr, d*, d)
        L  = self.L[site]          # (cl, wl, cl*)
        self.L[site + 1] = np.einsum(
            'ipl, ijk, lmnj, pnr -> rmr',
            L, A, W, A.conj(), optimize=True
        )

    # ------------------------------------------------------------------
    # Local 2-site eigensolver
    # ------------------------------------------------------------------
    def solve_local_2site(self, i, j):
        """
        Find the lowest-energy two-site wavefunction psi[chi_l, d, d, chi_r]
        by applying H_eff = L[i] * W[i] * W[j] * R[j+1] as a LinearOperator
        and calling ARPACK eigsh with which='SA'.
        Returns (psi_opt, energy).
        """
        L  = self.L[i]         # (cl, wl, cl*)
        R  = self.R[j + 1]     # (cr, wr, cr*)
        W1 = self.mpo[i]       # (wl, wm, d1*, d1)
        W2 = self.mpo[j]       # (wm, wr, d2*, d2)
        shape = (self.mps[i].shape[0], self.d, self.d, self.mps[j].shape[2])
        size  = int(np.prod(shape))

        def matvec(v):
            psi = v.reshape(shape)
            # Heff|psi> = L W1 W2 R |psi>
            res = np.einsum(
                'ipl, lmnj, mnok, qor, ijkr -> pqo',
                L, W1, W2, R, psi, optimize=True
            )
            return res.ravel()

        H_eff    = LinearOperator((size, size), matvec=matvec, dtype=float)
        psi_init = np.einsum('ijk,klm->ijlm', self.mps[i], self.mps[j], optimize=True).ravel()
        vals, vecs = eigsh(H_eff, k=1, which='SA', v0=psi_init, tol=1e-6, maxiter=1000)
        return vecs[:, 0].reshape(shape), float(vals[0])

    # ------------------------------------------------------------------
    # Truncation backends
    # ------------------------------------------------------------------
    def turboquant_truncate(self, mat, chi_target):
        """
        TurboQuant O(chi^2 log chi) truncation via FWHT basis rotation.
        Returns an approximate left-unitary matrix of shape (rows, chi_target).
        """
        rows, cols = mat.shape
        next_pow2  = 1 << (rows - 1).bit_length() if rows > 1 else 1
        mat_padded = np.pad(mat, ((0, next_pow2 - rows), (0, 0)))
        mat_rot    = fwht_vectorized(mat_padded)
        chi        = min(chi_target, cols, rows)
        norms      = np.linalg.norm(mat_rot, axis=1)  # score rows, not columns
        idx        = np.argsort(norms)[-chi:]
        res_rot    = mat_rot[idx, :]
        # Inverse FWHT on selected rows to reconstruct approximate subspace
        res_cols   = fwht_vectorized(res_rot.T)        # (cols, chi)
        res        = (res_cols / next_pow2).T          # (chi, cols)
        # QR to get left-unitary basis
        Q, _       = np.linalg.qr(res.T)              # Q shape: (cols, chi) or (rows-ish, chi)
        return Q                                        # (cols, chi)

    def svd_truncate(self, mat, chi_target):
        """Standard SVD truncation. Returns (U, S, Vh) truncated to chi_target."""
        U, S, Vh = np.linalg.svd(mat, full_matrices=False)
        chi      = min(chi_target, len(S))
        return U[:, :chi], S[:chi], Vh[:chi, :]

    # ------------------------------------------------------------------
    # Main sweep
    # ------------------------------------------------------------------
    def run_simulation(self, mode="svd", sweeps=3):
        """
        Run `sweeps` left-right DMRG sweeps.
        Returns (energies_per_site, mean_truncation_times) — one entry per sweep.
        """
        # Build all right environments before the first sweep
        for i in range(self.n_sites - 1, 0, -1):
            self.update_env_right(i)

        energies          = []
        truncation_times  = []

        for sweep in range(sweeps):
            sweep_times  = []
            last_energy  = 0.0

            # ---- Left → Right pass ----
            for i in range(self.n_sites - 1):
                psi_2site, e = self.solve_local_2site(i, i + 1)
                last_energy  = e
                cl, d1, d2, cr = psi_2site.shape
                mat = psi_2site.reshape(cl * d1, d2 * cr)

                t0 = time.perf_counter()
                if mode == "turboquant":
                    Q   = self.turboquant_truncate(mat, self.chi_max)
                    # Project: S·Vh = Q^T · mat
                    SV  = Q.T @ mat
                    chi_actual = Q.shape[1]
                    U_store    = Q
                    S_store    = np.ones(chi_actual)   # absorbed into SV
                    V_store    = SV
                else:
                    U_store, S_store, V_store = self.svd_truncate(mat, self.chi_max)
                    chi_actual = len(S_store)

                sweep_times.append(time.perf_counter() - t0)

                self.mps[i]     = U_store.reshape(cl, d1, chi_actual)
                self.mps[i + 1] = (np.diag(S_store) @ V_store).reshape(chi_actual, d2, cr)
                self.update_env_left(i)

            # ---- Right → Left pass ----
            for i in range(self.n_sites - 1, 0, -1):
                psi_2site, e = self.solve_local_2site(i - 1, i)
                last_energy  = e
                cl, d1, d2, cr = psi_2site.shape
                mat = psi_2site.reshape(cl * d1, d2 * cr)

                # Right pass always uses SVD to maintain right-canonical form
                U_store, S_store, V_store = self.svd_truncate(mat, self.chi_max)
                chi_actual = len(S_store)

                self.mps[i]     = V_store.reshape(chi_actual, d2, cr)
                self.mps[i - 1] = (U_store @ np.diag(S_store)).reshape(cl, d1, chi_actual)
                self.update_env_right(i)

            energies.append(float(last_energy / self.n_sites))
            truncation_times.append(float(np.mean(sweep_times)) if sweep_times else 0.0)

        return energies, truncation_times
