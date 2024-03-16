import numpy as np


def simplex_dual(c: np.ndarray, A: np.ndarray, b: np.ndarray, B: np.ndarray) -> np.ndarray:
    r"""Dual simplex method implementation."""
    m, n = A.shape

    while True:
        # Шаг 1 - Составим базисную матрицу и найдем обратную для нее
        A_B: np.ndarray = A[:, B]
        A_B_inv: np.ndarray = np.linalg.inv(A_B)

        # Шаг 2 - сформируем вектор, состоящий из компонент вектора с базисными индексами
        c_B: np.ndarray = c[B]

        # Шаг 3 - Находим базисный допустимый план двойственной задачи
        y: np.ndarray = c_B @ A_B_inv

        # Шаг 4 - Находим псевдоплан, соотв текущему базисному допустимому плану
        kk_B: np.ndarray = A_B_inv @ b
        kk = np.array([kk_B[np.where(B == i)][0] if i in B else 0 for i in range(n)])

        # Шаг 5
        if np.all(kk >= 0):
            return kk

        # Шаг 6 - Выделяем отрицательную компоненту псевдоплана и сохраняем ее индекс(он базисный)
        j_k: int = np.where(kk < 0)[0][-1]

        # Шаг 7 - пусть дельта у - к-ая строка обратной матрицы А
        k: int = np.where(B == j_k)[0][0]
        delta_y: np.ndarray = A_B_inv[k]
        mu = np.array([delta_y @ A[:, j] if j in np.setdiff1d(np.arange(n), B) else 0 for j in range(n)])

        # Шаг 8
        if np.all(mu[np.setdiff1d(np.arange(n), B)] >= 0):
            raise ValueError("Unable to find a solution!")

        # Шаг 9
        sigma = np.array([(c[j] - A[:, j] @ y) / mu[j] for j in np.setdiff1d(np.arange(n), B) if mu[j] < 0])

        # Шаг 10
        j_0: np.intp = np.argmin(sigma)
        # Шаг 11
        B[k] = j_0


if __name__ == "__main__":
    c = np.array([-4, -3, -7, 0, 0])
    A = np.array([
        [-2, -1, -4, 1, 0],
        [-2, -2, -2, 0, 1],
    ])
    b = np.array([-1, -3. / 2])
    B = np.array([3, 4])

    x: np.ndarray = simplex_dual(c=c, A=A, b=b, B=B)

    print(f"x = {x}")