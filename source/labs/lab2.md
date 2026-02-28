# Лабораторная работа №2: Работа с библиотеками NumPy, Pandas, Matplotlib и Seaborn
## Тихонова Роза 501455 P3122
## Описание проекта
Данная лабораторная работа демонстрирует основы работы с ключевыми библиотеками Python для научных вычислений и анализа данных:
- NumPy - для работы с многомерными массивами и матрицами
- Pandas - для загрузки и обработки данных
- Matplotlib - для базовой визуализации
- Seaborn - для статистической визуализации

## Запуск тестов:
**Запуск всех тестов с подробным выводом**
`python -m pytest test.py -v`

**Запуск конкретного теста**
`python -m pytest test.py::test_create_vector -v`

## Цель работы
- Научиться создавать и манипулировать многомерными массивами (векторами и матрицами)
- Освоить базовые операции линейной алгебры
- Научиться строить гистограммы для анализа распределения данных (Matplotlib)
- Освоить создание тепловых карт для визуализации корреляций (Seaborn)
- Научиться строить линейные графики для отображения зависимостей (Matplotlib)
## Задание
1. Создайте виртуальное окружение:
    `python -m venv numpy_env`
   
 2. Активируйте виртуальное окружение:
    - Windows: numpy_env\Scripts\activate
    - Linux/Mac: source numpy_env/bin/activate
   
 3. Установите зависимости:
    `pip install numpy matplotlib seaborn pandas pytest`

 Структура проекта:

 numpy_lab/
 ├── main.py
 ├── test.py
 ├── data/
 │   └── students_scores.csv
 └── plots/

 В папке data создайте файл students_scores.csv со следующим содержимым:

 math,physics,informatics
 78,81,90
 85,89,88
 92,94,95
 70,75,72
 88,84,91
 95,99,98
 60,65,70
 73,70,68
 84,86,85
 90,93,92

 - Задача: реализовать все функции, чтобы проходили тесты.
 
# Код

```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. СОЗДАНИЕ И ОБРАБОТКА МАССИВОВ

def create_vector():
    """
    Создать массив от 0 до 9.

    Изучить:
    https://numpy.org/doc/stable/reference/generated/numpy.arange.html
    
    Returns:
        numpy.ndarray: Массив чисел от 0 до 9 включительно
    """
    return np.arange(10)
    
   
def create_matrix():
    """
    Создать матрицу 5x5 со случайными числами [0,1].

    Изучить:
    https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html
    
    Returns:
        numpy.ndarray: Матрица 5x5 со случайными значениями от 0 до 1
    """
    return np.random.rand(5,5)

def reshape_vector(vec):
    """
    Преобразовать (10,) -> (2,5)

    Изучить:
    https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
    
    Args:
        vec (numpy.ndarray): Входной массив формы (10,)
    
    Returns:
        numpy.ndarray: Преобразованный массив формы (2, 5)
    """
    return vec.reshape(2,5)


def transpose_matrix(mat):
    """
    Транспонирование матрицы.

    Изучить:
    https://numpy.org/doc/stable/reference/generated/numpy.transpose.html
    
    Args:
        mat (numpy.ndarray): Входная матрица
    
    Returns:
        numpy.ndarray: Транспонированная матрица
    """
    return np.transpose(mat)
    

# 2. ВЕКТОРНЫЕ ОПЕРАЦИИ

def vector_add(a, b):
    """
    Сложение векторов одинаковой длины.
    (Векторизация без циклов)
    
    Args:
        a (numpy.ndarray): Первый вектор
        b (numpy.ndarray): Второй вектор
    
    Returns:
        numpy.ndarray: Результат поэлементного сложения
    """
    return a + b;


def scalar_multiply(vec, scalar):
    """
    Умножение вектора на число.
    
    Args:
        vec (numpy.ndarray): Входной вектор
        scalar (float/int): Число для умножения
    
    Returns:
        numpy.ndarray: Результат умножения вектора на скаляр
    """
    return vec*scalar


def elementwise_multiply(a, b):
    """
    Поэлементное умножение.
    
    Args:
        a (numpy.ndarray): Первый вектор/матрица
        b (numpy.ndarray): Второй вектор/матрица
    
    Returns:
        numpy.ndarray: Результат поэлементного умножения
    """
    return a*b;


def dot_product(a, b):
    """
    Скалярное произведение.

    Изучить:
    https://numpy.org/doc/stable/reference/generated/numpy.dot.html
    
    Args:
        a (numpy.ndarray): Первый вектор
        b (numpy.ndarray): Второй вектор
    
    Returns:
        float: Скалярное произведение векторов
    """
    return np.dot(a, b)


# 3. МАТРИЧНЫЕ ОПЕРАЦИИ

def matrix_multiply(a, b):
    """
    Умножение матриц.

    Изучить:
    https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
    
    Args:
        a (numpy.ndarray): Первая матрица
        b (numpy.ndarray): Вторая матрица
    
    Returns:
        numpy.ndarray: Результат умножения матриц
    """
    return np.matmul(a, b);


def matrix_determinant(a):
    """
    Определитель матрицы.

    Изучить:
    https://numpy.org/doc/stable/reference/generated/numpy.linalg.det.html
    
    Args:
        a (numpy.ndarray): Квадратная матрица
    
    Returns:
        float: Определитель матрицы
    """
    return np.linalg.det(a)


def matrix_inverse(a):
    """
    Обратная матрица.

    Изучить:
    https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html
    
    Args:
        a (numpy.ndarray): Квадратная матрица
    
    Returns:
        numpy.ndarray: Обратная матрица
    """
    return np.linalg.inv(a)


def solve_linear_system(a, b):
    """
    Решить систему Ax = b

    Изучить:
    https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html
    
    Args:
        a (numpy.ndarray): Матрица коэффициентов A
        b (numpy.ndarray): Вектор свободных членов b
    
    Returns:
        numpy.ndarray: Решение системы x
    """
    return np.linalg.solve(a, b)


# 4. СТАТИСТИЧЕСКИЙ АНАЛИЗ

def load_dataset(path="data/students_scores.csv"):
    """
    Загрузить CSV и вернуть NumPy массив.
    
    Args:
        path (str): Путь к CSV файлу
    
    Returns:
        numpy.ndarray: Загруженные данные в виде массива
    """
    return pd.read_csv(path).to_numpy()


def statistical_analysis(data):
     """
    Представьте, что данные — это результаты экзамена по математике.
    Нужно оценить:
    - средний балл
    - медиану
    - стандартное отклонение
    - минимум
    - максимум
    - 25 и 75 перцентили

    Изучить:
    https://numpy.org/doc/stable/reference/generated/numpy.mean.html
    https://numpy.org/doc/stable/reference/generated/numpy.median.html
    https://numpy.org/doc/stable/reference/generated/numpy.std.html
    https://numpy.org/doc/stable/reference/generated/numpy.percentile.html
    
    Args:
        data (numpy.ndarray): Одномерный массив данных
    
    Returns:
        dict: Словарь со статистическими показателями
    """
     return {
        "mean": np.mean(data),
        "median": np.median(data),
        "std": np.std(data),
        "min": np.min(data),
        "max": np.max(data),
        "percentile_25": np.percentile(data, 25),
        "percentile_75": np.percentile(data, 75)
    }


def normalize_data(data):
    """
    Min-Max нормализация.
    
    Формула: (x - min) / (max - min)
    
    Args:
        data (numpy.ndarray): Входной массив данных
    
    Returns:
        numpy.ndarray: Нормализованный массив данных в диапазоне [0, 1]
    """
    min_val = np.min(data)
    max_val = np.max(data)
    normalized = (data - min_val) / (max_val - min_val)
    return normalized

# 5. ВИЗУАЛИЗАЦИЯ

def plot_histogram(data):
    """
    Построить гистограмму распределения оценок по математике.

    Изучить:
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html
    
    Args:
        data (numpy.ndarray): Данные для гистограммы
    """
    plt.hist(data, bins='auto', alpha = 0.7)
    plt.title('Распределение оценок по математике')
    plt.xlabel('Оценки')
    plt.ylabel('Частота')
    plt.savefig('plots/histogram.png')
    pass


def plot_heatmap(matrix):
    """
    Построить тепловую карту корреляции предметов.

    Изучить:
    https://seaborn.pydata.org/generated/seaborn.heatmap.html
    
    Args:
        matrix (numpy.ndarray): Матрица корреляции
    """
    os.makedirs('plots', exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, 
                annot=True,           # Show values in cells
                cmap='coolwarm',      # Color map suitable for correlations (-1 to 1)
                center=0,             # Center the colormap at 0
                square=True,          # Make cells square
                cbar_kws={"shrink": 0.8}, # Shrink colorbar
                fmt='.2f',            # Format for annotations (2 decimal places)
                xticklabels=True,     # Show tick labels
                yticklabels=True)     # Show tick labels
    plt.title('Тепловая карта корреляции предметов', fontsize=14)
    plt.tight_layout()
    plt.savefig('plots/heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()



def plot_line(x, y):
    """
    Построить график зависимости: студент -> оценка по математике.

    Изучить:
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
    
    Args:
        x (numpy.ndarray): Номера студентов
        y (numpy.ndarray): Оценки студентов
    """
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker='o', linewidth=2, markersize=8, color='blue')
    
    # Добавляем заголовок и подписи осей
    plt.title('Зависимость оценок студентов от их номера', fontsize=14, fontweight='bold')
    plt.xlabel('Номер студента', fontsize=12)
    plt.ylabel('Оценка по математике', fontsize=12)
    
    plt.grid(True, alpha=0.3)
    
    plt.show()
```
## Выводы:
В ходе выполнения лабораторной работы №2 я узнала о ключевых библиотеках Python для научных вычислений: NumPy для работы с многомерными массивами, Pandas для загрузки табличных данных, Matplotlib и Seaborn для визуализации. Я научилась создавать векторы и матрицы различных размерностей, выполнять математические операции сложения и умножения.