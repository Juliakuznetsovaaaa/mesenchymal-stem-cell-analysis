import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


def plot_boxplots(file_paths, file_names=None, colors=None, save_dir=None, dpi=120):
    """
    Строит и сохраняет графики с боксплотами для каждого признака в данных
    со статистической значимостью различий между указанными парами групп.
    """
    # Загрузка данных и удаление лишних столбцов
    dfs = []
    for file in file_paths:
        df = pd.read_csv(file)
        # Удаляем ненужные столбцы
        df = df.drop(columns=['Label', ' '], errors='ignore')
        dfs.append(df)

    # Проверка совпадения признаков
    reference_columns = set(dfs[0].columns)
    for i, df in enumerate(dfs):
        if set(df.columns) != reference_columns:
            raise ValueError(f"Файл {file_paths[i]} имеет отличающиеся признаки")

    # Определение названий файлов
    if file_names is None:
        file_names = [os.path.basename(file) for file in file_paths]
    elif len(file_names) != len(file_paths):
        raise ValueError("Количество названий файлов должно совпадать с количеством файлов")

    # Цвета по умолчанию
    if colors is None:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    if len(colors) < len(file_paths):
        raise ValueError("Количество цветов должно быть не меньше количества файлов")

    # Создаем папку для сохранения если нужно
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Настройка русского шрифта
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # Определяем порядок групп для сравнения
    group_order = [
        'Здоровый контроль',
        'Здоровый с остеодифференцировкой',
        'Больной контроль',
        'Больной с остеодифференцировкой'
    ]

    # Сопоставляем переданные имена с нашим порядком
    group_indices = {name: idx for idx, name in enumerate(group_order)}
    sorted_indices = [group_indices[name] for name in file_names]

    # Определяем пары для сравнения
    comparison_pairs = [
        (0, 1),  # Здоровый контроль vs Здоровый с остеодифференцировкой
        (2, 3),  # Больной контроль vs Больной с остеодифференцировкой
        (1, 3),  # Здоровый с остеодиф. vs Больной с остеодиф.
        (0, 2)  # Здоровый контроль vs Больной контроль
    ]

    # Словарь для перевода названий признаков
    feature_translation = {
        'Area': 'Площадь',
        'Perim.': 'Периметр',
        # Можно добавить другие признаки при необходимости
    }

    # Для каждого признака строим отдельный график
    for feature in dfs[0].columns:
        plt.figure(figsize=(10, 8))

        # Собираем данные для текущего признака
        data_to_plot = []
        for i, df in enumerate(dfs):
            temp_df = pd.DataFrame({
                'Значение': df[feature],
                'Источник': file_names[i]
            })
            data_to_plot.append(temp_df)

        combined_data = pd.concat(data_to_plot, ignore_index=True)

        # Строим боксплоты с указанным порядком групп
        ax = sns.boxplot(
            x='Источник',
            y='Значение',
            data=combined_data,
            hue='Источник',
            palette=colors[:len(file_paths)],
            width=0.6,
            order=group_order,
            legend=False  # Отключаем легенду
        )

        feature_name = feature_translation.get(feature, feature)

        plt.title(f'Сравнение признака: {feature_name}', fontsize=14)
        plt.xlabel('Источник данных', fontsize=12)
        plt.ylabel(f'Значение ({feature_name})', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Рассчет статистической значимости для нужных пар
        group_data = [df[feature].values for df in dfs]

        # Собираем p-values для всех пар
        p_values = []
        for pair in comparison_pairs:
            i, j = pair
            _, p_val = mannwhitneyu(group_data[i], group_data[j], alternative='two-sided')
            p_values.append(p_val)

        # Применяем поправку Бенджамини-Хохберга
        reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

        # Функция для определения количества звезд
        def get_stars(p):
            if p < 0.0001:
                return '****'
            elif p < 0.001:
                return '***'
            elif p < 0.01:
                return '**'
            elif p < 0.05:
                return '*'
            else:
                return ''

        # Добавляем аннотации на график
        y_max = combined_data['Значение'].max()
        y_range = combined_data['Значение'].max() - combined_data['Значение'].min()

        # Текущая высота для линии
        h_level = y_max + 0.05 * y_range
        line_height_step = 0.08 * y_range

        # Аннотации для каждой пары
        for idx, pair in enumerate(comparison_pairs):
            if reject[idx]:
                # Определяем количество звезд по оригинальному p-value
                stars = get_stars(p_values[idx])

                # Координаты для линии
                x1, x2 = pair
                y = h_level

                # Рисуем линию
                plt.plot([x1, x1, x2, x2], [y, y + line_height_step / 2, y + line_height_step / 2, y],
                         lw=1.5, c='black')

                # Добавляем текст со звездами
                plt.text((x1 + x2) * 0.5, y + line_height_step / 2, stars,
                         ha='center', va='bottom', color='black', fontsize=12)

                # Увеличиваем высоту для следующей линии
                h_level += line_height_step

        # Обновляем пределы оси Y
        plt.ylim(top=h_level + line_height_step)

        plt.tight_layout()

        # Сохранение в файл если указана директория
        if save_dir:
            safe_feature = "".join([c if c.isalnum() else "_" for c in feature])
            save_path = os.path.join(save_dir, f'boxplot_{safe_feature}.png')
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"График сохранен: {save_path}")

        plt.show()
        plt.close()


# Пример использования
if __name__ == "__main__":
    files = [
        'full_dapi_bad_cntrl.csv',
        'full_dapi_bad_od.csv',
        'full_dapi_good_cntrl.csv',
        'full_dapi_good_od.csv'
    ]

    russian_names = [
        'Здоровый контроль',  # Индекс 0
        'Здоровый с остеодифференцировкой',  # Индекс 1
        'Больной контроль',  # Индекс 2
        'Больной с остеодифференцировкой'  # Индекс 3
    ]

    custom_colors = ['#FF5733', '#33FF57', '#3357FF', '#F333FF']

    plot_boxplots(
        file_paths=files,
        file_names=russian_names,
        colors=custom_colors,
        save_dir='boxplots_analysis',
        dpi=150
    )