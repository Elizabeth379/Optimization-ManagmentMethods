import numpy as np


def simplex_method(c, A, x, B):
    m, n = A.shape

    # Проверка на соответствие размеров
    assert len(c) == n, "Размер вектора c должен быть равен числу переменных"
    assert len(x) == n, "Размер вектора x должен быть равен числу переменных"
    assert len(B) == m, "Размер множества B должен быть равен числу ограничений"

    # Основной цикл симплекс-метода
    while True:
        B_matrix = A[:, B]
        print(B_matrix)
        B_inv = np.linalg.inv(B_matrix)
        c_B = c[B]

        # Вычисление вектора ценности
        delta_c = c - A.T.dot(B_inv.T).dot(c_B)

        # Проверка на оптимальность
        if np.all(delta_c >= 0):
            return x

        # Выбор ведущего столбца
        entering_index = np.where(delta_c < 0)[0][0]
        d = np.zeros(n)
        d[entering_index] = 1

        # Вычисление направления
        d_B = -B_inv.dot(A[:, entering_index])

        # Проверка на неограниченность
        if np.all(d_B >= 0):
            return "Целевая функция не ограничена сверху"

        # Вычисление длины шага
        ratios = np.divide(x[B], d_B, out=np.full_like(x[B], np.inf), where=d_B < 0)
        theta = np.min(ratios)

        # Обновление плана
        x -= theta * d_B
        x[abs(x) < 1e-10] = 0  # Очищаем значения, близкие к нулю

        # Обновление множества B
        leaving_index = np.argmin(ratios)
        B[leaving_index] = entering_index


if __name__ == '__main__':
    # Пример использования
    c = np.array([3, 2, 0, 0])
    A = np.array([[1, 0, 1, 0],
                  [0, 1, 0, 1]])
    x = np.array([1, 2, 0, 0])
    B = np.array([2, 3])  # Начальное множество базисных индексов

    optimal_plan = simplex_method(c, A, x, B)
    print("Оптимальный план:", optimal_plan)


