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
HIGH_QUBIT_START = TEST_N - 3
USE_QISKIT_RANDOM_TEST = True
QISKIT_RANDOM_N = 18
QISKIT_RANDOM_NUM_CNOTS = 100
QISKIT_RANDOM_SEED = 7
QISKIT_RANDOM_HIGH_QUBIT_START = QISKIT_RANDOM_N - 3
GF2 = galois.GF(2)


def print_cn_like_paper(c: NDArray[np.uint8]) -> None:
    c = GF2(c)
    n2 = c.shape[0]
    assert n2 % 2 == 0
    n = n2 // 2

    U = c[:n, :n]
    Z12 = c[:n, n:]
    Z21 = c[n:, :n]
    D = c[n:, n:]

    assert np.all(np.asarray(Z12, dtype=np.uint8) == 0)
    assert np.all(np.asarray(Z21, dtype=np.uint8) == 0)
    assert np.array_equal(
        np.asarray(D, dtype=np.uint8),
        np.asarray(np.linalg.inv(U.T), dtype=np.uint8),
    )

    print("U =")
    print(np.asarray(U, dtype=np.uint8))
    print("(U^T)^-1 =")
    print(np.asarray(D, dtype=np.uint8))


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

        # Canonicalize support positions so the lower-left block has the
        # Theorem 13 layout with first k rows/cols as the active support.
        c_arr = np.asarray(c_lower_left, dtype=np.uint8)
        nz_rows = [i for i in range(n) if np.any(c_arr[i, :])]
        nz_cols = [j for j in range(n) if np.any(c_arr[:, j])]
        if len(nz_rows) != k or len(nz_cols) != k:
            raise ValueError(
                f"Unexpected support in lower-left block: rows={len(nz_rows)}, cols={len(nz_cols)}, k={k}."
            )
        z_rows = [i for i in range(n) if i not in nz_rows]
        z_cols = [j for j in range(n) if j not in nz_cols]
        row_order = nz_rows + z_rows
        col_order = nz_cols + z_cols

        prow = GF2(np.eye(n, dtype=np.uint8)[row_order, :])
        pcol = GF2(np.eye(n, dtype=np.uint8)[:, col_order])
        left_perm = GF2(np.block([[prow, z], [z, prow]]))
        right_perm = GF2(np.block([[pcol, z], [z, pcol]]))

        m1 = left_perm @ m1 @ right_perm
        left_sigma = left_perm @ left_sigma
        right_tau = right_tau @ right_perm

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
            "left_sigma": np.asarray(left_sigma, dtype=np.uint8),
            "left_l": np.asarray(left_l, dtype=np.uint8),
            "right_u": np.asarray(right_u, dtype=np.uint8),
            "right_tau": np.asarray(right_tau, dtype=np.uint8),
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

        lc1, lp1, lc2, lp2 = lemma10(leftpc)
        rc1, rp1, rc2, rp2 = lemma10(rightpc)
        assert_in_cn_form(lc1)
        assert_in_cn_form(lc2)
        assert_in_cn_form(rc1)
        assert_in_cn_form(rc2)

        left_layer = GF2(lc1) @ GF2(lp1) @ GF2(lc2) @ GF2(lp2)
        right_layer = GF2(rc1) @ GF2(rp1) @ GF2(rc2) @ GF2(rp2)

        # Step 2/3 consistency: M3 = left_layer @ M1 @ right_layer.
        m1_from_m3 = np.linalg.inv(left_layer) @ GF2(m3) @ np.linalg.inv(right_layer)
        if not np.array_equal(
            np.asarray(m1_from_m3, dtype=np.uint8),
            np.asarray(step1["M1"], dtype=np.uint8),
        ):
            raise ValueError(
                "Theorem13 check failed: cannot recover M1 from M3 and Lemma10 layers."
            )

        # Full reconstruction of the original symplectic matrix.
        m_reconstructed = (
            np.linalg.inv(GF2(step1["left_l"]))
            @ np.linalg.inv(GF2(step1["left_sigma"]))
            @ m1_from_m3
            @ np.linalg.inv(GF2(step1["right_tau"]))
            @ np.linalg.inv(GF2(step1["right_u"]))
        )
        if not np.array_equal(
            np.asarray(m_reconstructed, dtype=np.uint8),
            np.asarray(self.symplectic_matrix, dtype=np.uint8),
        ):
            raise ValueError(
                "Theorem13 check failed: reconstructed matrix does not match original symplectic matrix."
            )

        # print_matrix_like_paper(lc1, [self.n, self.n], [self.n, self.n], name="lc1")
        # print_matrix_like_paper(lp1, [self.n, self.n], [self.n, self.n], name="lp1")

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


def print_matrix_like_paper(
    matrix: NDArray[np.uint8],
    row_block_sizes: list[int] | None = None,
    col_block_sizes: list[int] | None = None,
    name: str = "M",
) -> None:
    """
    Print a binary matrix in block form similar to paper notation.

    Example:
      print_matrix_like_paper(M, [n, n], [n, n], name="M")
    """
    arr = np.asarray(matrix, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError("Input must be a 2D matrix.")

    n_rows, n_cols = arr.shape
    if row_block_sizes is None:
        row_block_sizes = [n_rows]
    if col_block_sizes is None:
        col_block_sizes = [n_cols]

    if sum(row_block_sizes) != n_rows:
        raise ValueError("Sum of row_block_sizes must equal the number of rows.")
    if sum(col_block_sizes) != n_cols:
        raise ValueError("Sum of col_block_sizes must equal the number of columns.")

    row_cuts = np.cumsum(row_block_sizes)[:-1].tolist()
    col_cuts = np.cumsum(col_block_sizes)[:-1].tolist()

    print(f"{name}:")
    sep_width = 2 + (2 * n_cols - 1) + (3 * len(col_cuts))
    for i in range(n_rows):
        if i in row_cuts:
            print("-" * sep_width)

        blocks = []
        start = 0
        for cut in col_cuts + [n_cols]:
            block_vals = arr[i, start:cut]
            blocks.append(" ".join(str(int(v)) for v in block_vals))
            start = cut
        print("[ " + " | ".join(blocks) + " ]")


def swap_rows(a, i: int, k: int) -> None:
    a[[i, k], :] = a[[k, i], :]


def swap_cols(a, i: int, j: int) -> None:
    a[:, [i, j]] = a[:, [j, i]]


def row_xor(a, dst: int, src: int) -> None:
    a[dst, :] = a[dst, :] + a[src, :]


def col_xor(a, dst: int, src: int) -> None:
    a[:, dst] = a[:, dst] + a[:, src]


def _lemma7_lower(
    a_in: NDArray[np.uint8],
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """Lower-triangular form of Lemma 7: A = L L^T + Lambda over GF(2)."""
    a = GF2(a_in)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("A must be square.")
    if not np.array_equal(
        np.asarray(a, dtype=np.uint8), np.asarray(a.T, dtype=np.uint8)
    ):
        raise ValueError("A must be symmetric.")

    n = a.shape[0]
    l = identity(n)

    for i in range(1, n):
        for j in range(i):
            # A_ij = sum_{k<j} L_ik L_jk + L_ij  (since L_jj = 1)
            s = GF2(0)
            if j > 0:
                s = l[i, :j] @ l[j, :j].T
            l[i, j] = a[i, j] + s

    ll_t = l @ l.T
    diag_vals = np.diag(np.asarray(a + ll_t, dtype=np.uint8))
    lam = GF2.Zeros((n, n))
    for i in range(n):
        lam[i, i] = diag_vals[i]

    if not np.array_equal(
        np.asarray(ll_t + lam, dtype=np.uint8), np.asarray(a, dtype=np.uint8)
    ):
        raise ValueError("Lemma 7 decomposition failed sanity check.")

    return np.asarray(l, dtype=np.uint8), np.asarray(lam, dtype=np.uint8)


def optimize_for_pushing_cnot_target_to_lower(
    u_in: NDArray[np.uint8],
    m: int,
) -> NDArray[np.uint8]:
    """
    Placeholder optimization pass for CNOT synthesis ordering.
    Currently returns U unchanged.
    """
    print_matrix_like_paper(u_in, name="U")
    u = np.asarray(u_in, dtype=np.uint8)
    if u.ndim != 2 or u.shape[0] != u.shape[1]:
        raise ValueError("U must be square.")
    n = u.shape[0]
    if m < 0 or m > n:
        raise ValueError(f"m must be in [0, {n}], got {m}.")
    if m % 2 != 0:
        raise ValueError(f"m must be even to keep |J| even, got {m}.")
    high_qubit_start = HIGH_QUBIT_START
    if high_qubit_start < 0 or high_qubit_start > n:
        raise ValueError(
            f"HIGH_QUBIT_START must be in [0, {n}], got {high_qubit_start}."
        )

    # Bookkeeping state for greedy construction of J.
    j_set: set[int] = set()
    high_rows = list(range(high_qubit_start, n))
    k_r = {r: 0 for r in high_rows}
    v_r = {r: 0 for r in high_rows}

    bad_ones = count_bad_ones_for_high_rows(u, high_qubit_start)
    print(
        f"[optimize_for_pushing_cnot_target_to_lower] "
        f"high_qubit_start={high_qubit_start}, bad_ones={bad_ones}"
    )

    # Greedy construction of J up to size bound m.
    while len(j_set) < m:
        candidate_increase = {}
        for j in range(n):
            if j in j_set:
                continue
            candidate_increase[j] = increase_bad_ones(u, j, j_set, k_r, v_r, high_rows)

        if len(candidate_increase) == 0:
            break

        best_j, best_increase = min(candidate_increase.items(), key=lambda x: x[1])
        if best_increase >= 0 and (len(j_set) % 2 == 0):
            break

        j_set.add(best_j)
        for r in high_rows:
            u_rj = int(u[r, best_j] & 1)
            k_r[r] += u_rj
            v_r[r] ^= u_rj

    if len(j_set) == 0:
        print("no optimize")
        return u

    if len(j_set) % 2 != 0:
        raise ValueError(
            "Constructed |J| is odd; cannot form T = I + uu^T with TT^T=I."
        )

    u_selector = GF2.Zeros((n, 1))
    for j in j_set:
        u_selector[j, 0] = 1

    t = identity(n) + (u_selector @ u_selector.T)
    s = GF2(u) @ t
    s_np = np.asarray(s, dtype=np.uint8)

    print_matrix_like_paper(s_np, name="S")
    return s_np


def increase_bad_ones(
    u_in: NDArray[np.uint8],
    j: int,
    j_set: set[int],
    k_r: dict[int, int],
    v_r: dict[int, int],
    high_rows: list[int],
) -> int:
    """
    Compute increase of bad_ones if column j is added to current J.
    """
    u = np.asarray(u_in, dtype=np.uint8)
    if u.ndim != 2 or u.shape[0] != u.shape[1]:
        raise ValueError("U must be square.")
    n = u.shape[0]
    if j < 0 or j >= n:
        raise ValueError(f"j must be in [0, {n-1}], got {j}.")
    if j in j_set:
        raise ValueError(f"j={j} is already in J.")

    new_j_size = len(j_set) + 1
    increase = 0
    for r in high_rows:
        u_rj = int(u[r, j] & 1)
        new_k_r = k_r[r] + u_rj
        new_v_r = v_r[r] ^ u_rj
        if new_v_r == 1:
            increase += new_j_size - new_k_r
    return increase


def count_bad_ones_for_high_rows(u_in: NDArray[np.uint8], high_qubit_start: int) -> int:
    """
    Count bad ones for a general matrix using high-row cost:
      bad_ones = (# of ones on high rows) - (# of high rows).
    """
    u = np.asarray(u_in, dtype=np.uint8)
    if u.ndim != 2 or u.shape[0] != u.shape[1]:
        raise ValueError("U must be square.")
    n = u.shape[0]
    if high_qubit_start < 0 or high_qubit_start > n:
        raise ValueError(
            f"high_qubit_start must be in [0, {n}], got {high_qubit_start}."
        )

    high_row_ones = int(np.sum(u[high_qubit_start:, :] & 1))
    num_high_rows = n - high_qubit_start
    return high_row_ones - num_high_rows


def lemma10(
    a_in: NDArray[np.uint8],
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """
    Lemma 10 decomposition over GF(2): A = U U^T + Lambda,
    where U is invertible upper-triangular and Lambda is diagonal.
    """
    a = GF2(a_in)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("A must be square.")
    if not np.array_equal(
        np.asarray(a, dtype=np.uint8), np.asarray(a.T, dtype=np.uint8)
    ):
        raise ValueError("A must be symmetric.")

    n = a.shape[0]
    p = GF2(np.eye(n, dtype=np.uint8)[::-1])  # reversal permutation
    a_rev = p @ a @ p.T

    l_rev, lam_rev = _lemma7_lower(np.asarray(a_rev, dtype=np.uint8))
    l_rev = GF2(l_rev)
    lam_rev = GF2(lam_rev)

    u = p @ l_rev @ p.T
    u = GF2(
        optimize_for_pushing_cnot_target_to_lower(np.asarray(u, dtype=np.uint8), m=4)
    )
    lam = p @ lam_rev @ p.T

    # print_matrix_like_paper(u, name="U")

    u_np = np.asarray(u, dtype=np.uint8)
    if not np.all(u_np[np.tril_indices(n, k=-1)] == 0):
        raise ValueError("Lemma 10 check failed: U is not upper triangular.")
    if not np.all(np.diag(u_np) == 1):
        raise ValueError("Lemma 10 check failed: U is not unit diagonal.")
    lam_np = np.asarray(lam, dtype=np.uint8)
    if np.any(lam_np - np.diag(np.diag(lam_np))):
        raise ValueError("Lemma 10 check failed: Lambda is not diagonal.")
    if not np.array_equal(
        np.asarray(u @ u.T + lam, dtype=np.uint8), np.asarray(a, dtype=np.uint8)
    ):
        raise ValueError("Lemma 10 check failed: A != U U^T + Lambda.")

    # Corollary 11 check for input a_in as the symmetric upper-right block B:
    # [[I, B], [0, I]] = C-P-C-P factors from U and Lambda.
    i_n = identity(n)
    z_n = GF2.Zeros((n, n))
    c1 = GF2(np.block([[u, z_n], [z_n, np.linalg.inv(u.T)]]))
    p1 = GF2(np.block([[i_n, i_n], [z_n, i_n]]))
    c2 = GF2(np.block([[np.linalg.inv(u), z_n], [z_n, u.T]]))
    p2 = GF2(np.block([[i_n, lam], [z_n, i_n]]))
    four_factor = c1 @ p1 @ c2 @ p2
    target = GF2(np.block([[i_n, a], [z_n, i_n]]))
    if not np.array_equal(
        np.asarray(four_factor, dtype=np.uint8), np.asarray(target, dtype=np.uint8)
    ):
        raise ValueError(
            "Corollary 11 check failed: C-P-C-P does not match [[I, A], [0, I]]."
        )

    return c1, p1, c2, p2


def assert_in_cn_form(c_in: NDArray[np.uint8]) -> None:
    c = GF2(c_in)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or c.shape[0] % 2 != 0:
        raise ValueError("C-matrix check failed: input must be 2n x 2n.")

    n2 = c.shape[0]
    n = n2 // 2

    a = c[:n, :n]
    z12 = c[:n, n:]
    z21 = c[n:, :n]
    d = c[n:, n:]

    if np.any(np.asarray(z12, dtype=np.uint8)):
        raise ValueError("C-matrix check failed: top-right block is not zero.")
    if np.any(np.asarray(z21, dtype=np.uint8)):
        raise ValueError("C-matrix check failed: bottom-left block is not zero.")

    a_inv_t = np.linalg.inv(a).T
    if not np.array_equal(
        np.asarray(d, dtype=np.uint8), np.asarray(a_inv_t, dtype=np.uint8)
    ):
        raise ValueError("C-matrix check failed: D != (A^{-1})^T.")

    a_np = np.asarray(a, dtype=np.uint8)
    if not np.all(a_np[np.tril_indices(n, k=-1)] == 0):
        raise ValueError("C-matrix check failed: A is not upper triangular.")

    d_np = np.asarray(d, dtype=np.uint8)
    if not np.all(d_np[np.triu_indices(n, k=1)] == 0):
        raise ValueError("C-matrix check failed: D is not lower triangular.")


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


def random_symplectic_matrix_from_qiskit_cnot(
    n: int,
    num_cnots: int,
    high_qubit_start: int,
    seed: int | None = None,
) -> NDArray[np.uint8]:
    """
    Build a random CNOT-only Clifford circuit in Qiskit and return its
    2n x 2n symplectic matrix over GF(2).

    Targets are restricted to qubits in [high_qubit_start, n-1].
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if num_cnots < 0:
        raise ValueError("num_cnots must be non-negative.")
    if high_qubit_start < 0 or high_qubit_start >= n:
        raise ValueError(
            f"high_qubit_start must be in [0, {n-1}], got {high_qubit_start}."
        )

    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Clifford
    except ImportError as exc:
        raise ImportError(
            "Qiskit is required for random_symplectic_matrix_from_qiskit_cnot."
        ) from exc

    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for _ in range(num_cnots):
        target = int(rng.integers(high_qubit_start, n))
        control = int(rng.integers(0, n - 1))
        if control >= target:
            control += 1
        qc.cx(control, target)

    s = np.asarray(Clifford(qc).symplectic_matrix, dtype=np.uint8)
    if not check_symplectic(s, convention="standard"):
        raise ValueError("Qiskit-derived symplectic matrix failed check_symplectic.")
    return s


if __name__ == "__main__":
    if USE_QISKIT_RANDOM_TEST:
        S = random_symplectic_matrix_from_qiskit_cnot(
            n=QISKIT_RANDOM_N,
            num_cnots=QISKIT_RANDOM_NUM_CNOTS,
            high_qubit_start=QISKIT_RANDOM_HIGH_QUBIT_START,
            seed=QISKIT_RANDOM_SEED,
        )
        print_matrix_like_paper(S, name="S")
    else:
        S = fixed_symplectic_matrix()
    assert check_symplectic(S, convention="standard")
    MaslovRoettelerNF(S).theorem13()
