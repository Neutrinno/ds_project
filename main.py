from typing import List
from ds_helper import mean, median, std


def main() -> None:
    """
    Точка входа в приложение.
    Демонстрация использования функций из ds_helper.
    """
    sample_data: List[float] = [10, 20, 30, 40, 50]

    print(">>> Running stats from ds_helper")
    print(f"Data: {sample_data}")
    print(f"Mean: {mean(sample_data)}")
    print(f"Median: {median(sample_data)}")
    print(f"Std: {std(sample_data)}")


if __name__ == "__main__":
    main()
