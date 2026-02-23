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
# TEST_I = 3
# Full-rank C example:
TEST_I = 174533521197412727885233567186830402475
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
        int, NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]
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
            k,
            np.asarray(l, dtype=np.uint8),
            np.asarray(u, dtype=np.uint8),
            np.asarray(sigma, dtype=np.uint8),
            np.asarray(tau, dtype=np.uint8),
        )

    def theorem13_step1(self):
        # Step 1: Apply Lemma 12.
        k, l, u, sigma, tau = self.lemma12()

        n = self.n
        m = GF2(self.symplectic_matrix)

        l_mat = GF2(l)
        u_mat = GF2(u)
        sigma_mat = GF2(sigma)
        tau_mat = GF2(tau)
        z = GF2.Zeros((n, n))

        left_sigma = GF2(np.block([[sigma_mat, z], [z, sigma_mat]]))
        left_l = GF2(
            np.block([[l_mat.T, z], [z, np.linalg.inv(l_mat)]])
        )  # row-convention equivalent of diag((L^T)^-1, L)
        right_u = GF2(
            np.block([[np.linalg.inv(u_mat), z], [z, u_mat.T]])
        )  # row-convention equivalent of diag(U, (U^T)^-1)
        right_tau = GF2(np.block([[tau_mat, z], [z, tau_mat]]))  # diag(tau, tau)

        m1 = left_sigma @ left_l @ m @ right_u @ right_tau

        c_lower_left = m1[n:, :n]
        k_from_m1 = int(np.count_nonzero(np.any(c_lower_left.row_reduce(), axis=1)))
        if k_from_m1 != k:
            raise ValueError(
                f"Inconsistent k: lemma12 returned {k}, but M1 implies {k_from_m1}."
            )

        r0 = slice(0, k)
        r1 = slice(k, n)
        r2 = slice(n, n + k)
        c0 = slice(0, k)
        c1 = slice(k, n)
        c2 = slice(n, n + k)
        c3 = slice(n + k, 2 * n)

        a1 = m1[r0, c0]
        a2 = m1[r0, c1]
        a3 = m1[r1, c0]
        a4 = m1[r1, c1]
        b1 = m1[r0, c2]
        b2 = m1[r0, c3]
        b3 = m1[r1, c2]
        b4 = m1[r1, c3]
        d1 = m1[r2, c2]
        d2 = m1[r2, c3]

        if np.any(np.asarray(a2, dtype=np.uint8)):
            raise ValueError("Theorem13 step1 check failed: A2 is not zero.")
        if not np.array_equal(
            np.asarray(a4, dtype=np.uint8), np.asarray(identity(n - k), dtype=np.uint8)
        ):
            raise ValueError("Theorem13 step1 check failed: A4 is not I_(n-k).")
        if not np.array_equal(
            np.asarray(a1, dtype=np.uint8), np.asarray(a1.T, dtype=np.uint8)
        ):
            raise ValueError("Theorem13 step1 check failed: A1 is not symmetric.")

        return {
            "k": k,
            "M1": np.asarray(m1, dtype=np.uint8),
            "A1": np.asarray(a1, dtype=np.uint8),
            "A2": np.asarray(a2, dtype=np.uint8),
            "A3": np.asarray(a3, dtype=np.uint8),
            "A4": np.asarray(a4, dtype=np.uint8),
            "B1": np.asarray(b1, dtype=np.uint8),
            "B2": np.asarray(b2, dtype=np.uint8),
            "B3": np.asarray(b3, dtype=np.uint8),
            "B4": np.asarray(b4, dtype=np.uint8),
            "D1": np.asarray(d1, dtype=np.uint8),
            "D2": np.asarray(d2, dtype=np.uint8),
        }

    def theorem13(self):
        step1 = self.theorem13_step1()
        leftpc, step2 = self.theorem13_step2(step1)
        rightpc, m3 = self.theorem13_step3(step2)
        assert leftpc.shape == (self.n, self.n)
        assert rightpc.shape == (self.n, self.n)
        assert np.array_equal(
            np.asarray(leftpc, dtype=np.uint8), np.asarray(leftpc.T, dtype=np.uint8)
        )
        assert np.array_equal(
            np.asarray(rightpc, dtype=np.uint8), np.asarray(rightpc.T, dtype=np.uint8)
        )

        print(leftpc)
        print(rightpc)

    def theorem13_step2(self, step1_result):
        n = self.n
        k = int(step1_result["k"])
        m1 = GF2(step1_result["M1"])
        a1 = GF2(step1_result["A1"])
        a3 = GF2(step1_result["A3"])

        a_sym = GF2.Zeros((n, n))
        a_sym[:k, :k] = a1
        a_sym[:k, k:] = a3.T
        a_sym[k:, :k] = a3
        # lower-right block stays zero

        i_k = identity(k)
        i_nk = identity(n - k)
        z_k_nk = GF2.Zeros((k, n - k))
        z_nk_k = GF2.Zeros((n - k, k))
        z_nk_nk = GF2.Zeros((n - k, n - k))
        z_k_k = GF2.Zeros((k, k))

        left_step2 = GF2(
            np.block(
                [
                    [i_k, z_k_nk, a1, a3.T],
                    [z_nk_k, i_nk, a3, z_nk_nk],
                    [z_k_k, z_k_nk, i_k, z_k_nk],
                    [z_nk_k, z_nk_nk, z_nk_k, i_nk],
                ]
            )
        )
        m2 = left_step2 @ m1

        r0 = slice(0, k)
        r1 = slice(k, n)
        r2 = slice(n, n + k)
        c2 = slice(n, n + k)
        c3 = slice(n + k, 2 * n)

        b1p = m2[r0, c2]
        b2p = m2[r0, c3]
        b3p = m2[r1, c2]
        b4p = m2[r1, c3]
        d1 = m2[r2, c2]
        d2 = m2[r2, c3]

        if not np.array_equal(
            np.asarray(b1p, dtype=np.uint8), np.asarray(identity(k), dtype=np.uint8)
        ):
            raise ValueError("Theorem13 step2 check failed: B1' is not I_k.")
        if not np.array_equal(
            np.asarray(d1, dtype=np.uint8), np.asarray(d1.T, dtype=np.uint8)
        ):
            raise ValueError("Theorem13 step2 check failed: D1 is not symmetric.")
        if np.any(np.asarray(b2p, dtype=np.uint8)):
            raise ValueError("Theorem13 step2 check failed: B2' is not zero.")
        if not np.array_equal(
            np.asarray(b4p, dtype=np.uint8), np.asarray(b4p.T, dtype=np.uint8)
        ):
            raise ValueError("Theorem13 step2 check failed: B4' is not symmetric.")
        if not np.array_equal(
            np.asarray(b3p, dtype=np.uint8), np.asarray(d2.T, dtype=np.uint8)
        ):
            raise ValueError("Theorem13 step2 check failed: B3' is not equal to D2^T.")

        return np.asarray(a_sym, dtype=np.uint8), {
            "M2": np.asarray(m2, dtype=np.uint8),
            "B1p": np.asarray(b1p, dtype=np.uint8),
            "B2p": np.asarray(b2p, dtype=np.uint8),
            "B3p": np.asarray(b3p, dtype=np.uint8),
            "B4p": np.asarray(b4p, dtype=np.uint8),
            "D1": np.asarray(d1, dtype=np.uint8),
            "D2": np.asarray(d2, dtype=np.uint8),
        }

    def theorem13_step3(self, step2_result):
        n = self.n
        m2 = GF2(step2_result["M2"])
        d1 = GF2(step2_result["D1"])
        d2 = GF2(step2_result["D2"])
        b4p = GF2(step2_result["B4p"])

        k = d1.shape[0]
        if d1.shape != (k, k):
            raise ValueError(f"Invalid D1 shape {d1.shape}.")
        if d2.shape != (k, n - k):
            raise ValueError(f"Invalid D2 shape {d2.shape}; expected ({k}, {n-k}).")
        if b4p.shape != (n - k, n - k):
            raise ValueError(f"Invalid B4' shape {b4p.shape}; expected ({n-k}, {n-k}).")

        sym_step3 = GF2.Zeros((n, n))
        sym_step3[:k, :k] = d1
        sym_step3[:k, k:] = d2
        sym_step3[k:, :k] = d2.T
        sym_step3[k:, k:] = b4p

        i_k = identity(k)
        i_nk = identity(n - k)
        z_k_nk = GF2.Zeros((k, n - k))
        z_nk_k = GF2.Zeros((n - k, k))
        z_nk_nk = GF2.Zeros((n - k, n - k))
        z_k_k = GF2.Zeros((k, k))

        right_step3 = GF2(
            np.block(
                [
                    [i_k, z_k_nk, d1, d2],
                    [z_nk_k, i_nk, d2.T, b4p],
                    [z_k_k, z_k_nk, i_k, z_k_nk],
                    [z_nk_k, z_nk_nk, z_nk_k, i_nk],
                ]
            )
        )

        m3 = m2 @ right_step3

        return np.asarray(sym_step3, dtype=np.uint8), np.asarray(m3, dtype=np.uint8)


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
        np.asarray(a @ b.T, dtype=np.uint8), np.asarray(b @ a.T, dtype=np.uint8)
    ):
        raise ValueError(
            "Input blocks must satisfy C D^T = D C^T over GF(2) in row convention."
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
    MaslovRoettelerNF(S).theorem13()
