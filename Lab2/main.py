import numpy as np


def simplex_method(c, x, A, b, B):
    m, n = A.shape

    x_new = x.copy()

    assert (np.linalg.matrix_rank(A) == m)

    # Основной цикл симплекс-метода
    while True:
        #Шаг 1 - строим базисную матрицу и находим обратную
        B_matrix = A[:, B]
        B_inv = np.linalg.inv(B_matrix)

        #Шаг 2 - вектор компонент вектора С
        c_B = c[B]

        # Шаг 3 - вектор потенциалов
        u = c_B @ B_inv
        # Шаг 4 - вектор оценок
        delta_c = u @ A - c

        # Шаг 5 - Проверка на оптимальность
        if np.all(delta_c >= 0):
            assert np.all(A @ x == b)
            return x_new

        # Шаг 6 - находим первую отрицат компоненту
        j0 = np.where(delta_c < 0)[0][0]

        # Шаг 7 - базисную обратную умножаем на столбец матрицы А с индексом j0
        z = B_inv @ A[:, j0]

        # Шаг 8
        theta = np.array([x_new[B[i]] / z[i] if z[i] > 0 else np.inf for i in range(m)])

        # Шаг 9
        theta0 = np.min(theta)

        # Шаг 10
        if theta0 == np.inf:
            raise Exception("Целевая функция не ограничена сверху на множестве допустимых планов")

        # Шаг 11
        k = np.argmin(theta)
        j_star = B[k]

        # Шаг 12
        B[k] = j0

        # Шаг 13
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

    optimal_plan = simplex_method(c, x, A, b, B)
    print("Оптимальный план:", optimal_plan)


