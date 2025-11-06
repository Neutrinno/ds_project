
def mean(numbers):
    """
    Вычисляет среднее значение списка чисел
    """
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)

def median(numbers):
    if not numbers:
        return 0
    s = sorted(numbers)
    n = len(s)
    mid = n // 2
    return (s[mid - 1] + s[mid]) / 2 if n % 2 == 0 else s[mid]

def std(numbers):
    """
    Простое выборочное стандартное отклонение
    """
    if not numbers:
        return 0
    m = mean(numbers)
    var = sum((x - m) ** 2 for x in numbers) / (len(numbers) - 1) if len(numbers) > 1 else 0
    return var ** 0.5

if __name__ == "__main__":
    sample_data = [10, 20, 30, 40, 50]
    print(">>> ds_helper: basic stats (from feature/add-std)")
    print(f"Data: {sample_data}")
    print(f"Mean: {mean(sample_data)}")
    print(f"Median: {median(sample_data)}")
    print(f"Std: {std(sample_data)}")

