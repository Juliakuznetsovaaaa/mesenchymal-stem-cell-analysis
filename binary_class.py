import pandas as pd
import numpy as np
import matplotlib
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import compute_class_weight

# Общая константа для random_state
RANDOM_STATE = 52

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay, auc
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import make_pipeline
from collections import Counter
import joblib
import json
import datetime
import os
from scipy.stats import randint, uniform, loguniform
from sklearn.base import clone


# 1. Загрузка данных с бинарной классификацией
def load_data_binary():
    dfs = []
    for file, class_name in [
        ('full_dapi_good_cntrl.csv', 'healthy'),
        ('full_dapi_bad_cntrl.csv', 'disease'),
    ]:
        df = pd.read_csv(file)
        df['class'] = class_name
        cols_to_drop = [col for col in ['Label', ' ', 'target'] if col in df.columns]
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True).convert_dtypes()
    return data


data = load_data_binary()

# 2. Предобработка
non_numeric_cols = data.select_dtypes(exclude=np.number).columns.difference(['class'])
if not non_numeric_cols.empty:
    print(f"Удаление нечисловых столбцов: {list(non_numeric_cols)}")
    data = data.drop(columns=non_numeric_cols)
data = data.loc[:, ~data.columns.duplicated()]
data = data.replace([np.inf, -np.inf], np.nan).dropna()

# 3. Кодирование бинарных меток
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['class'])  # 0=healthy, 1=disease
X = data.drop('class', axis=1)
balancer = RandomUnderSampler(
    sampling_strategy=1.0,
    random_state=RANDOM_STATE
)
X, y = balancer.fit_resample(X, y)


# 4. Выбор признаков
def select_top_features(X, y, n_features=8):
    selector = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, class_weight='balanced')
    selector.fit(X, y)
    importances = selector.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_features = X.columns[indices[:n_features]]

    print(f"\nТоп-{n_features} важных признаков:")
    for i, feature in enumerate(top_features, 1):
        print(f"{i}. {feature}: {importances[indices[i - 1]]:.4f}")

    return top_features


top_features = select_top_features(X, y, n_features=8 )
X = X[top_features]

print(f"\nВсего признаков: {X.shape[1]}")
print(f"Всего образцов: {X.shape[0]}")
print("Распределение классов:")
class_dist = pd.Series(y).value_counts()
print(class_dist)
print(f"Кодировка классов: {label_encoder.classes_} [0=healthy, 1=disease]")

# 5. Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print("\nРаспределение классов в обучающей выборке:")
print(pd.Series(y_train).value_counts())


# 6. Улучшенная функция оценки для бинарной классификации
def evaluate_binary_model(model, model_name, X_train, y_train, X_test, y_test, results_dir, class_names):
    # Оценка на тестовых данных
    y_pred_test = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test)[:, 1]

    # Оценка на тренировочных данных
    y_pred_train = model.predict(X_train)
    y_proba_train = model.predict_proba(X_train)[:, 1]

    # Основные метрики (тест)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    roc_auc_test = roc_auc_score(y_test, y_proba_test)

    # Основные метрики (тренировка)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    roc_auc_train = roc_auc_score(y_train, y_proba_train)

    print(f"\n{model_name} Metrics:")
    print(f"Test Accuracy: {accuracy_test:.4f} | Train Accuracy: {accuracy_train:.4f}")
    print(f"Test ROC-AUC: {roc_auc_test:.4f} | Train ROC-AUC: {roc_auc_train:.4f}")

    # Детальные отчеты
    report_test = classification_report(y_test, y_pred_test, target_names=class_names, output_dict=True)
    report_train = classification_report(y_train, y_pred_train, target_names=class_names, output_dict=True)

    print("\nTest Classification Report:")
    print(pd.DataFrame(report_test).transpose())
    print("\nTrain Classification Report:")
    print(pd.DataFrame(report_train).transpose())

    # Сохранение метрик
    metrics = {
        'test_accuracy': accuracy_test,
        'test_roc_auc': roc_auc_test,
        'test_classification_report': report_test,
        'train_accuracy': accuracy_train,
        'train_roc_auc': roc_auc_train,
        'train_classification_report': report_train
    }

    # Матрица ошибок (тест)
    plt.figure(figsize=(8, 6))
    cm_test = confusion_matrix(y_test, y_pred_test)
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Test Confusion Matrix - {model_name}')
    plt.ylabel('Истинные метки')
    plt.xlabel('Предсказанные метки')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/{model_name}_test_confusion_matrix.png')
    plt.close()

    # Матрица ошибок (тренировка)
    plt.figure(figsize=(8, 6))
    cm_train = confusion_matrix(y_train, y_pred_train)
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Train Confusion Matrix - {model_name}')
    plt.ylabel('Истинные метки')
    plt.xlabel('Предсказанные метки')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/{model_name}_train_confusion_matrix.png')
    plt.close()

    # ROC-кривая (тест)
    plt.figure(figsize=(8, 6))
    RocCurveDisplay.from_predictions(y_test, y_proba_test, name=f'{model_name} (Test)')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'Test ROC Curve - {model_name} (AUC = {roc_auc_test:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig(f'{results_dir}/{model_name}_test_roc_curve.png')
    plt.close()

    # ROC-кривая (тренировка)
    plt.figure(figsize=(8, 6))
    RocCurveDisplay.from_predictions(y_train, y_proba_train, name=f'{model_name} (Train)')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'Train ROC Curve - {model_name} (AUC = {roc_auc_train:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig(f'{results_dir}/{model_name}_train_roc_curve.png')
    plt.close()

    # Важность признаков
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(10, 6))
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.title(f'Feature Importance - {model_name}')
        plt.bar(range(X.shape[1]), importances[indices], align='center')
        plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
        plt.tight_layout()
        plt.savefig(f'{results_dir}/{model_name}_feature_importance.png')
        plt.close()

    return metrics


# 7. Функция для расчета весов классов
def calculate_scale_pos_weight(y):
    class_counts = np.bincount(y)
    return class_counts[0] / class_counts[1]


# 8. Обучение моделей с бинарной классификацией
def train_and_evaluate_binary(X_train, y_train, X_test, y_test, results_dir, class_names):
    results = {}
    best_models = {}

    # Веса классов
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

    # Конфигурация базовых моделей
    base_models = [
        ('rf', make_pipeline(
            RandomForestClassifier(random_state=RANDOM_STATE, class_weight=class_weight_dict)
        ), {
             'randomforestclassifier__n_estimators': randint(200, 800),
             'randomforestclassifier__max_depth': [None, 15, 25, 35],
             'randomforestclassifier__min_samples_split': randint(2, 15),
             'randomforestclassifier__min_samples_leaf': randint(1, 8)
         }),
        ('logreg', make_pipeline(
            StandardScaler(),
            LogisticRegression(random_state=RANDOM_STATE, class_weight='balanced')
        ), {
             'logisticregression__C': loguniform(1e-3, 1e3),
             'logisticregression__penalty': ['l1', 'l2'],
             'logisticregression__solver': ['liblinear']
         }),

        ('cat', make_pipeline(
            CatBoostClassifier(random_state=RANDOM_STATE, verbose=0, auto_class_weights='Balanced')
        ), {
             'catboostclassifier__iterations': randint(300, 1000),
             'catboostclassifier__learning_rate': loguniform(1e-3, 0.3),
             'catboostclassifier__depth': randint(6, 12),
             'catboostclassifier__l2_leaf_reg': loguniform(1e-3, 10)
         }),

        ('svm', make_pipeline(
            StandardScaler(),
            SVC(random_state=RANDOM_STATE, class_weight='balanced', probability=True)
        ), {
             'svc__C': loguniform(1e-3, 1e3),
             'svc__kernel': ['linear', 'rbf', 'poly'],
             'svc__gamma': ['scale', 'auto']
         })
    ]

    # Обучение базовых моделей
    for name, pipeline, params in base_models:
        print("\n" + "=" * 60)
        print(f"Обучение базовой модели {name} с подбором гиперпараметров")
        print("=" * 60)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=params,
            n_iter=20,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,  # Используем все ядра
            verbose=1,
            random_state=RANDOM_STATE
        )

        search.fit(X_train, y_train)
        best_models[name] = search.best_estimator_

        print(f"Лучшие параметры {name}:")
        print(search.best_params_)

        # Оценка модели
        metrics = evaluate_binary_model(
            best_models[name],
            name,
            X_train, y_train,
            X_test, y_test,
            results_dir,
            class_names
        )
        results[name] = {
            'test_accuracy': metrics['test_accuracy'],
            'test_roc_auc': metrics['test_roc_auc'],
            'params': search.best_params_
        }

    # Добавляем StackingClassifier
    print("\n" + "=" * 60)
    print("Обучение StackingClassifier")
    print("=" * 60)

    # Лучшие версии базовых моделей
    estimators = [
        ('rf', best_models['rf']),
        ('cat', best_models['cat']),
        ('svm', best_models['svm'])
    ]

    # Мета-модель
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(class_weight='balanced', random_state=RANDOM_STATE),
        cv=5,  # Стратифицированная кросс-валидация
        n_jobs=-1,
        stack_method='auto'
    )

    # Обучение стекинга
    stacking.fit(X_train, y_train)

    # Оценка
    stacking_metrics = evaluate_binary_model(
        stacking,
        'stacking',
        X_train, y_train,
        X_test, y_test,
        results_dir,
        class_names
    )
    results['stacking'] = {
        'test_accuracy': stacking_metrics['test_accuracy'],
        'test_roc_auc': stacking_metrics['test_roc_auc']
    }

    # Сохраняем stacking модель
    best_models['stacking'] = stacking

    return best_models, results


# 9. Создание директории для результатов
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"binary_results_{timestamp}"
os.makedirs(results_dir, exist_ok=True)

# 10. Запуск обучения
best_models, results = train_and_evaluate_binary(
    X_train, y_train,
    X_test, y_test,
    results_dir,
    label_encoder.classes_
)

# 11. Сохранение результатов
for name, model in best_models.items():
    joblib.dump(model, f'{results_dir}/{name}_model.pkl')

# Сохранение метрик
with open(f'{results_dir}/metrics.json', 'w') as f:
    json.dump(results, f, indent=4)

# 12. Финальный отчет
print("\n" + "=" * 60)
print("ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ БИНАРНОЙ КЛАССИФИКАЦИИ")
print("=" * 60)
for name, res in results.items():
    print(
        f"{name.upper():<10} | Test Accuracy: {res['test_accuracy']:.4f} | Train Accuracy: {res['train_accuracy']:.4f}")
    print(f"{' ':<10} | Test ROC-AUC: {res['test_roc_auc']:.4f} | Train ROC-AUC: {res['train_roc_auc']:.4f}")
    print("-" * 60)
print("=" * 60)

# Дополнительная информация
data_info = {
    'features': list(X.columns),
    'class_distribution': {
        'healthy': class_dist.get(0, 0),
        'disease': class_dist.get(1, 0)
    },
    'preprocessing': 'StandardScaler + RandomUnderSampler',
    'target_mapping': dict(zip([0, 1], label_encoder.classes_)),
    'random_state': RANDOM_STATE
}
with open(f'{results_dir}/data_info.json', 'w') as f:
    json.dump(data_info, f, indent=4)