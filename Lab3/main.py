import numpy as np
from lab2_simplex import simplex_main


def simplex_init(c, A, b):
    m, n = A.shape

    # Шаг 1 - - делаем вектор правых частей b неотрицательным
    b[b < 0] *= -1
    A[b < 0] *= -1

    # Шаг 2 - составляем вспомагательную задачу линейного программирования
    c_tilde = np.hstack((np.zeros(n), np.full((m,), -1)))
    A_tilde = np.hstack((A, np.eye(m)))

    # Шаг 3 - Строим начальный базисный допустимый план
    x_tilde = np.concatenate((np.zeros(n), b))
    B = np.arange(n, n + m)

    # Шаг 4 - решаем вспомогательную задачу из шага 2 основной фазой симплекс-метода
    x_tilde, B = simplex_main(c_tilde, x_tilde, A_tilde, b, B)

    # Шаг 5 - проверяем условия совместности
    if np.any(x_tilde[n:n+m] != 0):
        raise ValueError("The task is unsolvable!")

    # Шаг 6 - формируем допустимый план
    x = x_tilde[0:n]

    while True:
        # Шаг 7 - если В... то метод завершает работу и возвращает допустимый план
        if set(B).issubset(set(np.arange(n))):
            return x, B, A, b

        # Шаг 8 - Выбираем в наборе В макс индекс искусственной переменной
        j_k = np.max(B[B in np.arange(n, n + m)])
        i = j_k - n
        k = np.where(B == j_k)

        # Шаг
        A_B =  np.linalg.inv(A_tilde[:, B])
        l = np.array([A_B @ A_tilde[:, j] for j in np.setdiff1d(np.arange(n), B)])

        if np.any(np.array([l_i[k] for l_i in l]) != 0):
            # Step 10
            j = np.where([l_i[k] for l_i in l]) != 0
            B[k] = j
        else:
            # Step 11
            A = np.delete(A, i, axis=0)
            A_tilde = np.delete(A_tilde, i, axis=0)
            b = np.delete(b, i, axis=0)
            B = np.delete(B, k, axis=0)



if __name__ == '__main__':
    c = np.array([1, 0, 0])
    A = np.array([
        [1, 1, 1],
        [2, 2, 2],
    ])
    b = np.array([0, 0])

    result = simplex_init(c.copy(), A.copy(), b.copy())

    print("Basis plan is", *zip(("x", "B", "A", "b"), result), sep="\n")