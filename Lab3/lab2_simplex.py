import numpy as np


def simplex_main(c, x, A, b, B):
    m, n = A.shape
    x_new = x.copy()

    assert (np.linalg.matrix_rank(A) == m)

    while True:
        # Step 1
        AB = A[:, B]
        A_inv_B = np.linalg.inv(AB)

        # Step 2
        c_B = c[B]

        # Step 3
        u = c_B @ A_inv_B

        # Step 4
        delta = u @ A - c

        # Step 5
        if np.all(delta >= 0):
            assert np.all(A @ x == b)
            return x_new, B

        # Step 6
        j0 = np.where(delta < 0)[0][0]

        # Step 7
        z = A_inv_B @ A[:, j0]

        # Step 8
        theta = np.array([x_new[B[i]] / z[i] if z[i] > 0 else np.inf for i in range(m)])

        # Step 9
        theta0 = np.min(theta)

        # Step 10
        if theta0 == np.inf:
            raise Exception("Целевая функция не ограничена сверху на множестве допустимых планов")

        # Step 11
        k = np.argmin(theta)
        j_star = B[k]

        # Step 12
        B[k] = j0

        # Step 12
        for i in range(m):
            if i != k:
                x_new[B[i]] -= theta0 * z[i]

        x_new[j0] = theta0
        x_new[j_star] = 0


if __name__ == '__main__':
    c = np.array([1, 1, 0, 0, 0])
    x = np.array([0, 0, 1, 3, 2])
    A = np.array([
        [-1, 1, 1, 0, 0],
        [1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1]
    ])
    b = np.array([1, 3, 2])
    B = np.array([2, 3, 4])

    result = simplex_main(c, x, A, b, B)

    print("Optimal plan is: ", *result, sep="\n")