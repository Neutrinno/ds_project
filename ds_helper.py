from typing import List


def mean(numbers: List[float]) -> float:
    """
    Вычисляет среднее значение списка чисел
    """
    if not numbers:
        return 0.0
    return float(sum(numbers)) / len(numbers)


def median(numbers: List[float]) -> float:
    """
    Вычисляет медиану списка чисел
    """
    if not numbers:
        return 0.0

    s = sorted(numbers)
    n = len(s)
    mid = n // 2

    return float((s[mid - 1] + s[mid]) / 2) if n % 2 == 0 else float(s[mid])


def std(numbers: List[float]) -> float:
    """
    Простое выборочное стандартное отклонение
    """
    if not numbers:
        return 0.0

    m = mean(numbers)
    var = (
        sum((x - m) ** 2 for x in numbers) / (len(numbers) - 1)
        if len(numbers) > 1
        else 0.0
    )
    return float(var ** 0.5)


if __name__ == "__main__":
    sample_data: List[float] = [10, 20, 30, 40, 50]

    print(">>> ds_helper: basic stats (from feature/add-std)")
    print(f"Data: {sample_data}")
    print(f"Mean: {mean(sample_data)}")
    print(f"Median: {median(sample_data)}")
    print(f"Std: {std(sample_data)}")
