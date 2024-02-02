import numpy as np


if __name__ == '__main__':
    n = int(input("Square matrix size: "))
    A = np.random.randint(-10, 10, size=(n, n))
    x = np.random.randint(-10, 10, size=(n,))

    i = int(input("Index of the matrix column to replace: "))
    assert(i < n)

    A_dash = A.copy().T
    A_dash[i] = x
    A_dash = A_dash.T
    A_inv = np.linalg.inv(A)

    l = A_inv @ x

    if l[i]==0:
        raise Exception("Matrix A_dash is irreversible")

    l_tilde = l.copy()
    l_tilde[i] = -1

    l_hat = (-1 / l[i]) * l_tilde

    Q = np.eye(n)
    Q = Q.T
    Q[i] = l_hat
    Q = Q.T

    A_dash_inv = Q @ A_inv

    print("Original matrix A: \n", A, "\n")
    print(f"Matrix A with column #{i} replaced with {x} (A_dash): \n", A_dash, "\n")
    print("Inverse matrix A_dash: \n", A_dash_inv, "\n")
    print("Inverse matrix A_dash with numpy (validation): \n", np.linalg.inv(A_dash), "\n")