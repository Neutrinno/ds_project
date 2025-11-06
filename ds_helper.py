
def mean(numbers):
    """
    Вычисляет среднее значение списка чисел
    """
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)


if __name__ == "__main__":
    sample_data = [10, 20, 30, 40, 50]
    avg = mean(sample_data)
    print(f"Sample data: {sample_data}")
    print(f"Mean value: {avg}")
