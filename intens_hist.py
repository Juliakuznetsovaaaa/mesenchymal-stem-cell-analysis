import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from scipy.stats import gamma, lognorm, norm


def mannwhitneyu_bootstrap(data1, data2, n_iterations=10000, alpha=0.05):
    """
    Выполняет бутстрап-тест Манна-Уитни для сравнения двух выборок.
    """
    # 1. Вычисляем наблюдаемую статистику
    mann_whitney = stats.mannwhitneyu(data1, data2)
    observed_statistic = mann_whitney.statistic

    # 2. Объединяем выборки
    combined_data = np.concatenate([data1, data2])
    n1, n2 = len(data1), len(data2)

    # 3. Бутстрап-итерации
    bootstrap_stats = []
    for _ in range(n_iterations):
        resample = np.random.choice(combined_data, size=n1 + n2, replace=True)
        resample1 = resample[:n1]
        resample2 = resample[n1:]

        # Вычисление статистики с правильными параметрами
        u_stat = stats.mannwhitneyu(resample1, resample2,
                                    alternative='two-sided').statistic
        bootstrap_stats.append(u_stat)

    # 4. Корректный расчет p-значения (двусторонний)
    # Вариант 1: Простой подсчет экстремальных значений
    extreme_count = np.sum(np.abs(bootstrap_stats) >= np.abs(observed_statistic))
    p_value = (extreme_count + 1) / (n_iterations + 1)  # С поправкой на непрерывность

    # Вариант 2: Смещение распределения (предпочтительный)
    # shifted_stats = bootstrap_stats - np.mean(bootstrap_stats)
    # p_value = np.mean(np.abs(shifted_stats) >= np.abs(observed_statistic - np.mean(bootstrap_stats)))

    return {
        'p_value': p_value,
        'observed_statistic': observed_statistic,
        'bootstrap_statistics': bootstrap_stats
    }


def hist(df1, df2, name, title):
    data1 = df1[name].dropna()
    data2 = df2[name].dropna()

    # Вычисление статистик
    results = mannwhitneyu_bootstrap(data1, data2, n_iterations=100)
    mann_whitney = stats.mannwhitneyu(data1, data2)
    p_value = mann_whitney.pvalue
    bootstrap_p = results['p_value']

    # Сбор текстовой информации
    stats_text = (
        "Статистика Манна-Уитни:\n"
        f"• Наблюдаемая статистика: {results['observed_statistic']:.2f}\n"
        f"• P-value (обычный): {p_value}\n"
        f"• P-value (бутстрап): {bootstrap_p:.4f}\n\n"

        "Описательная статистика:\n"
        "Остеодифференцированные:\n"
        f"• Среднее: {np.mean(data1):.2f}\n"
        f"• Медиана: {np.median(data1):.2f}\n"
        f"• Ст.отклонение: {np.std(data1):.2f}\n\n"

        "Контрольные:\n"
        f"• Среднее: {np.mean(data2):.2f}\n"
        f"• Медиана: {np.median(data2):.2f}\n"
        f"• Ст.отклонение: {np.std(data2):.2f}\n\n"

    )

    # Создание графиков
    plt.figure(figsize=(14, 6))

    # Левая панель - текстовая статистика
    ax_left = plt.subplot(1, 2, 1)
    ax_left.text(0.05, 0.95, stats_text, ha='left', va='top',
                 fontfamily='monospace', fontsize=14, transform=ax_left.transAxes)
    ax_left.axis('off')

    # Правая панель - процентная гистограмма
    plt.subplot(1, 2, 2)

    weights1 = np.ones_like(data1) / len(data1) * 100
    weights2 = np.ones_like(data2) / len(data2) * 100

    plt.hist(data1, weights=weights1, alpha=0.7, label='Остеодифференцированные')
    plt.hist(data2, weights=weights2, alpha=0.7, label='Контрольные')

    plt.grid(True, alpha=0.4)
    plt.title("Процентное распределение")
    plt.xlabel(title)
    plt.ylabel("Процент, %")
    plt.legend()

    plt.tight_layout()
    plt.show()
if __name__ == '__main__':
        file1='merged_file_cntrl_without_mut.csv'
        file2='merged_file_od_without_mut.csv'
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)

        print("=====mean=====")
        hist(df1, df2, 'Area', "Площадь")
        print("=====mean=====")
        hist(df1, df2, 'Perim.', "Периметр")
        print("=====mean=====")
        hist(df1, df2, 'Circ.', "Округлость")