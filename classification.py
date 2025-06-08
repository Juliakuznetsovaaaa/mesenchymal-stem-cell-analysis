import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, fbeta_score
from imblearn.under_sampling import RandomUnderSampler
import joblib
import shap

# Конфигурация
RANDOM_STATE = 47
TEST_SIZE = 0.1
DATA_PATHS = {
    'control': 'full_dapi_good_od.csv',
    'case': 'full_dapi_bad_od.csv'
}


def load_and_balance():
    """Загрузка и полная балансировка данных"""
    # Загрузка данных
    df_control = pd.read_csv(DATA_PATHS['control']).assign(target=0)
    df_case = pd.read_csv(DATA_PATHS['case']).assign(target=1)

    # Объединение и очистка
    data = pd.concat([df_control, df_case], axis=0)
    data = data.drop(columns=['Label', ' '], errors='ignore')
    data = data.loc[:, ~data.columns.duplicated()]

    # Удаление некорректных значений
    data = data.replace([np.inf, -np.inf], np.nan).dropna()


    # Балансировка всего датасета
    X = data.drop(columns=['target', 'RawIntDen'])

    y = data['target']

    balancer = RandomUnderSampler(
        sampling_strategy=1.0,  # Точный баланс
    )
    return balancer.fit_resample(X, y)


def prepare_datasets(X, y):
    """Подготовка стратифицированных наборов данных"""
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )


def train_pipeline(params):
    """Создание тренировочного пайплайна"""
    return Pipeline([
        ('scaler', RobustScaler()),
        ('classifier', CatBoostClassifier(
            **params,
            random_state=RANDOM_STATE,
            thread_count=1,
            verbose=False
        ))
    ])


def objective(trial, X_train, y_train):
    """Функция оптимизации гиперпараметров"""
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
        'depth': trial.suggest_int('depth', 3, 8),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10.0),
        'iterations': trial.suggest_int('iterations', 100, 500),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
    }

    model = train_pipeline(params)

    score = cross_val_score(
        model,
        X_train,
        y_train,
        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
        scoring='accuracy',
        n_jobs=-1
    ).mean()

    return score


if __name__ == "__main__":
    # Полная балансировка данных
    X_balanced, y_balanced = load_and_balance()

    # Проверка баланса
    print("\nСбалансированный датасет:")
    print(pd.Series(y_balanced).value_counts())

    # Стратифицированное разделение после балансировки
    X_train, X_test, y_train, y_test = prepare_datasets(X_balanced, y_balanced)

    # Контроль размеров
    print(f"\nРазмеры данных:")
    print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    print(f"Соотношение Train/Test: {1 - TEST_SIZE:.0%}/{TEST_SIZE:.0%}")

    # Оптимизация гиперпараметров
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train),
                   n_trials=100, show_progress_bar=True)

    # Обучение финальной модели
    best_model = train_pipeline(study.best_params)
    best_model.fit(X_train, y_train)

    # Сохранение модели
    joblib.dump(best_model, 'fully_balanced_model.pkl')


    # Функция оценки
    def evaluate(model, X, y, name):
        y_pred = model.predict(X)
        print(f"\n{name} Результаты:")
        print(f"AUC-ROC: {roc_auc_score(y, model.predict_proba(X)[:, 1]):.4f}")
        print(classification_report(y, y_pred))
        sns.heatmap(confusion_matrix(y, y_pred), annot=True, fmt='d')
        plt.title(f'Confusion Matrix ({name})')
        plt.show()


    # Оценка на сбалансированных данных
    evaluate(best_model, X_train, y_train, 'Balanced Train')
    evaluate(best_model, X_test, y_test, 'Balanced Test')

    # Интерпретация модели
    explainer = shap.TreeExplainer(best_model.named_steps['classifier'])
    shap_values = explainer.shap_values(best_model['scaler'].transform(X_train))
    shap.summary_plot(shap_values, X_train, feature_names=X_balanced.columns)