from __future__ import annotations

import numpy as np
import galois
from numpy.typing import NDArray
from pygsti.tools.symplectic import (
    change_symplectic_form_convention,
    check_symplectic,
    compute_symplectic_matrix,
)

TEST_N = 8
TEST_I = 3
# Full-rank C example:
# TEST_I = 174533521197412727885233567186830402475
GF2 = galois.GF(2)


class MaslovRoettelerNF:
    def __init__(self, symplectic_matrix: NDArray[np.uint8]) -> None:
        matrix = np.asarray(symplectic_matrix, dtype=np.uint8)

        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Input must be a square matrix.")
        if matrix.shape[0] % 2 != 0:
            raise ValueError("Input must have even dimension 2n x 2n.")
        if not check_symplectic(matrix, convention="standard"):
            raise ValueError("Input matrix is not symplectic (standard convention).")

        self.symplectic_matrix = matrix
        self.n = matrix.shape[0] // 2

    def lemma12(
        self,
    ) -> tuple[
        NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]
    ]:
        m = self.symplectic_matrix[self.n :, :]
        l1, p1, m1, u1, u1inv_t = lpu_rect_symplectic(m)
        k, l2_small, sigma_small, l2p_small, tau_small = second_stage_lpl_from_P1_M1(
            np.asarray(p1, dtype=np.uint8),
            np.asarray(m1, dtype=np.uint8),
        )

        p1_arr = np.asarray(p1, dtype=np.uint8)
        r = [i for i in range(self.n) if np.all(p1_arr[i, :] == 0)]
        c = [j for j in range(self.n) if np.all(p1_arr[:, j] == 0)]

        l2 = identity(self.n)
        sigma_stage = identity(self.n)
        l2p = identity(self.n)
        tau_stage = identity(self.n)
        if len(r) > 0:
            l2[np.ix_(r, r)] = GF2(l2_small)
            sigma_stage[np.ix_(r, r)] = GF2(sigma_small)
            l2p[np.ix_(c, c)] = GF2(l2p_small)
            tau_stage[np.ix_(c, c)] = GF2(tau_small)

        diag_u1 = GF2.Zeros((2 * self.n, 2 * self.n))
        diag_u1[: self.n, : self.n] = GF2(u1)
        diag_u1[self.n :, self.n :] = GF2(u1inv_t)

        diag_l2p = GF2.Zeros((2 * self.n, 2 * self.n))
        diag_l2p[: self.n, : self.n] = np.linalg.inv(GF2(l2p)).T
        diag_l2p[self.n :, self.n :] = GF2(l2p)

        diag_tau = GF2.Zeros((2 * self.n, 2 * self.n))
        diag_tau[: self.n, : self.n] = GF2(tau_stage)
        diag_tau[self.n :, self.n :] = GF2(tau_stage)

        final_block_matrix = (
            GF2(sigma_stage)
            @ GF2(l2)
            @ GF2(l1)
            @ GF2(m)
            @ diag_u1
            @ diag_l2p
            @ diag_tau
        )
        left_block = final_block_matrix[:, : self.n]
        right_block = final_block_matrix[:, self.n :]

        nz_rows = [
            i
            for i in range(self.n)
            if np.any(np.asarray(left_block[i, :], dtype=np.uint8))
        ]
        nz_cols = [
            j
            for j in range(self.n)
            if np.any(np.asarray(left_block[:, j], dtype=np.uint8))
        ]
        if len(nz_rows) != k or len(nz_cols) != k:
            raise ValueError(
                f"Unexpected left-block support: expected k={k}, got rows={len(nz_rows)}, cols={len(nz_cols)}."
            )

        zero_rows = [i for i in range(self.n) if i not in nz_rows]
        zero_cols = [j for j in range(self.n) if j not in nz_cols]
        d1 = right_block[np.ix_(nz_rows, nz_cols)]
        d2 = right_block[np.ix_(nz_rows, zero_cols)]
        z1 = right_block[np.ix_(zero_rows, nz_cols)]
        i2 = right_block[np.ix_(zero_rows, zero_cols)]
        if d1.shape != (k, k):
            raise ValueError(
                f"Unexpected D1 shape: got {d1.shape}, expected ({k}, {k})."
            )
        if d2.shape != (k, self.n - k):
            raise ValueError(
                f"Unexpected D2 shape: got {d2.shape}, expected ({k}, {self.n - k})."
            )
        if z1.shape != (self.n - k, k):
            raise ValueError(
                f"Unexpected lower-left right-block shape: got {z1.shape}, expected ({self.n - k}, {k})."
            )
        if i2.shape != (self.n - k, self.n - k):
            raise ValueError(
                f"Unexpected lower-right right-block shape: got {i2.shape}, expected ({self.n - k}, {self.n - k})."
            )

        left_11 = left_block[np.ix_(nz_rows, nz_cols)]
        left_12 = left_block[np.ix_(nz_rows, zero_cols)]
        left_21 = left_block[np.ix_(zero_rows, nz_cols)]
        left_22 = left_block[np.ix_(zero_rows, zero_cols)]

        if not np.array_equal(
            np.asarray(left_11, dtype=np.uint8),
            np.asarray(identity(k), dtype=np.uint8),
        ):
            raise ValueError("Final block check failed: top-left block is not I_k.")
        if np.any(np.asarray(left_12, dtype=np.uint8)):
            raise ValueError(
                "Final block check failed: top-right block of left half is not zero."
            )
        if np.any(np.asarray(left_21, dtype=np.uint8)):
            raise ValueError(
                "Final block check failed: bottom-left block of left half is not zero."
            )
        if np.any(np.asarray(left_22, dtype=np.uint8)):
            raise ValueError(
                "Final block check failed: bottom-right block of left half is not zero."
            )
        if np.any(np.asarray(z1, dtype=np.uint8)):
            raise ValueError(
                "Final block check failed: lower-left block of right half is not zero."
            )
        if not np.array_equal(
            np.asarray(i2, dtype=np.uint8),
            np.asarray(identity(self.n - k), dtype=np.uint8),
        ):
            raise ValueError(
                "Final block check failed: lower-right block of right half is not I_(n-k)."
            )

        l = np.linalg.inv(GF2(l1)) @ np.linalg.inv(GF2(l2))
        sigma = np.linalg.inv(GF2(sigma_stage))
        tau = np.linalg.inv(GF2(tau_stage))
        u = GF2(l2p).T @ np.linalg.inv(GF2(u1))

        diag_tau_out = GF2.Zeros((2 * self.n, 2 * self.n))
        diag_tau_out[: self.n, : self.n] = GF2(tau)
        diag_tau_out[self.n :, self.n :] = GF2(tau)

        diag_u_out = GF2.Zeros((2 * self.n, 2 * self.n))
        diag_u_out[: self.n, : self.n] = GF2(u)
        diag_u_out[self.n :, self.n :] = np.linalg.inv(GF2(u)).T

        recovered_m = (
            GF2(l) @ GF2(sigma) @ GF2(final_block_matrix) @ diag_tau_out @ diag_u_out
        )
        if not np.array_equal(
            np.asarray(recovered_m, dtype=np.uint8), np.asarray(m, dtype=np.uint8)
        ):
            raise ValueError(
                "Final decomposition check failed: M != L sigma B diag(tau,tau) diag(U,(U^{-1})^T)."
            )

        return (
            np.asarray(l, dtype=np.uint8),
            np.asarray(u, dtype=np.uint8),
            np.asarray(sigma, dtype=np.uint8),
            np.asarray(tau, dtype=np.uint8),
        )

    def theorem13(self) -> None:
        # Step 1: Apply Lemma 12.
        l, u, sigma, tau = self.lemma12()


def identity(n: int):
    return GF2(np.eye(n, dtype=np.uint8))


def swap_rows(a, i: int, k: int) -> None:
    a[[i, k], :] = a[[k, i], :]


def swap_cols(a, i: int, j: int) -> None:
    a[:, [i, j]] = a[:, [j, i]]


def row_xor(a, dst: int, src: int) -> None:
    a[dst, :] = a[dst, :] + a[src, :]


def col_xor(a, dst: int, src: int) -> None:
    a[:, dst] = a[:, dst] + a[:, src]


def lpu_rect_symplectic(
    m: NDArray[np.uint8],
):
    m = GF2(m)
    n, two_n = m.shape
    if two_n != 2 * n:
        raise ValueError("Input must be n x 2n.")

    a = m[:, :n].copy()
    b = m[:, n:].copy()
    if not np.array_equal(
        np.asarray(a.T @ b, dtype=np.uint8), np.asarray(b.T @ a, dtype=np.uint8)
    ):
        raise ValueError(
            "Input blocks must satisfy C^T D = D^T C over GF(2). Check step 1 in theorem 13"
        )

    sigma = identity(n)
    l1 = identity(n)

    u = identity(n)
    uinv_t = identity(n)

    r = 0
    for c_idx in range(n):
        if r >= n:
            break

        pivot = None
        for i in range(r, n):
            if a[i, c_idx] == 1:
                pivot = i
                break
        if pivot is None:
            continue

        if pivot != r:
            swap_rows(a, pivot, r)
            swap_rows(b, pivot, r)
            swap_rows(sigma, pivot, r)
            swap_rows(l1, pivot, r)

        for i in range(r + 1, n):
            if a[i, c_idx] == 1:
                row_xor(a, i, r)
                row_xor(b, i, r)
                row_xor(l1, i, r)

        for j in range(c_idx + 1, n):
            if a[r, j] == 1:
                col_xor(a, j, c_idx)
                col_xor(u, j, c_idx)

                col_xor(b, c_idx, j)
                col_xor(uinv_t, c_idx, j)

        r += 1

    p1 = a
    m1 = b

    pm = GF2.Zeros((n, 2 * n))
    pm[:, :n] = p1
    pm[:, n:] = m1

    right = GF2.Zeros((2 * n, 2 * n))
    right[:n, :n] = u
    right[n:, n:] = uinv_t

    reconstructed = np.linalg.inv(l1) @ pm @ np.linalg.inv(right)
    assert np.array_equal(
        np.asarray(reconstructed, dtype=np.uint8),
        np.asarray(m, dtype=np.uint8),
    ), "Sanity check failed: M != L1^{-1}[P1|M1]diag(U1,(U1^{-1})^T)^{-1}."

    return l1, p1, m1, u, uinv_t


def lpl_decompose_to_identity_gf2(
    s_in: NDArray[np.uint8],
):
    """
    LPL-style decomposition over GF(2) for a square matrix S (m x m).
    Uses row swaps + row XOR and column swaps + column XOR to reduce S to I.

    Returns:
      L2, sigma, L2p, tau, S_reduced

    Such that:  (sigma @ L2) @ S_in @ (L2p @ tau) == I   over GF(2).

    Notes:
      - sigma and tau track swaps only.
      - L2 and L2p are swap-free elimination factors (XOR only).
    """
    s = GF2(s_in).copy()
    if s.ndim != 2 or s.shape[0] != s.shape[1]:
        raise ValueError("S must be square.")
    m = s.shape[0]

    sigma = identity(m)
    tau = identity(m)

    left_total = identity(m)
    right_total = identity(m)

    for k in range(m):
        piv_i = piv_j = None
        found = False
        for i in range(k, m):
            for j in range(k, m):
                if s[i, j] == 1:
                    piv_i, piv_j = i, j
                    found = True
                    break
            if found:
                break

        if not found:
            raise ValueError(
                "Submatrix is singular over GF(2); cannot reduce to identity."
            )

        if piv_i != k:
            swap_rows(s, piv_i, k)
            swap_rows(sigma, piv_i, k)
            swap_rows(left_total, piv_i, k)

        if piv_j != k:
            swap_cols(s, piv_j, k)
            swap_cols(tau, piv_j, k)
            swap_cols(right_total, piv_j, k)

        for i in range(m):
            if i != k and s[i, k] == 1:
                row_xor(s, i, k)
                row_xor(left_total, i, k)

        for j in range(m):
            if j != k and s[k, j] == 1:
                col_xor(s, j, k)
                col_xor(right_total, j, k)

    l2 = left_total @ sigma.T
    l2p = right_total @ tau.T

    return (
        np.asarray(l2, dtype=np.uint8),
        np.asarray(sigma, dtype=np.uint8),
        np.asarray(l2p, dtype=np.uint8),
        np.asarray(tau, dtype=np.uint8),
        np.asarray(s, dtype=np.uint8),
    )


def second_stage_lpl_from_P1_M1(
    p1: NDArray[np.uint8],
    m1: NDArray[np.uint8],
):
    """
    Implements the paper's second step:
      - compute R (zero rows of P1) and C (zero cols of P1)
      - restrict S = M1[R, C]
      - do LPL on S to get sigma2, tau2, L2, L2'
    Returns (k, L2, sigma2, L2p, tau2).
    """
    p1_arr = np.asarray(p1, dtype=np.uint8)
    m1_arr = np.asarray(m1, dtype=np.uint8)
    n = p1_arr.shape[0]
    if p1_arr.shape != (n, n) or m1_arr.shape != (n, n):
        raise ValueError("P1 and M1 must both be n x n.")

    r = [i for i in range(n) if np.all(p1_arr[i, :] == 0)]
    c = [j for j in range(n) if np.all(p1_arr[:, j] == 0)]

    p1_rank = int(np.count_nonzero(np.any(GF2(p1_arr).row_reduce(), axis=1)))
    expected = n - p1_rank
    if len(r) != expected or len(c) != expected:
        raise ValueError(
            f"Inconsistent supports: expected |R|=|C|={expected} from rank(P1)={p1_rank}, "
            f"got |R|={len(r)}, |C|={len(c)}."
        )

    m = len(r)
    if m == 0:
        # second-stage LPL is vacuous
        l2 = identity(0)
        sigma2 = identity(0)
        l2p = identity(0)
        tau2 = identity(0)
        return p1_rank, l2, sigma2, l2p, tau2

    s = m1_arr[np.ix_(r, c)]
    l2, sigma2, l2p, tau2, sred = lpl_decompose_to_identity_gf2(s)

    if not np.array_equal(sred, np.asarray(identity(m), dtype=np.uint8)):
        raise ValueError(
            "Unexpected: restricted block did not reduce to identity; check assumptions."
        )

    return p1_rank, l2, sigma2, l2p, tau2


def fixed_symplectic_matrix() -> NDArray[np.uint8]:
    """Return the fixed 2n x 2n symplectic matrix used for testing."""
    s = change_symplectic_form_convention(compute_symplectic_matrix(TEST_I, TEST_N))
    return np.asarray(s, dtype=np.uint8)


S = fixed_symplectic_matrix()


if __name__ == "__main__":
    assert check_symplectic(S, convention="standard")
    MaslovRoettelerNF(S).lemma12()
