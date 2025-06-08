import os
import cv2
import numpy as np
import networkx as nx
from skimage.morphology import skeletonize
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


def image_to_graph(image_path, debug=False):
    """Преобразование изображения в граф с проверкой на каждом этапе"""
    # Загрузка изображения
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Ошибка загрузки изображения: {image_path}")
        return None

    # Проверка размера изображения
    if img.size == 0:
        print(f"Пустое изображение: {image_path}")
        return None

    # Адаптивная бинаризация
    bin_img = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Морфологическая обработка
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    processed = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)

    # Скелетизация
    skeleton = skeletonize(processed // 255).astype(np.uint8) * 255

    # Визуализация для отладки
    if debug:
        plt.figure(figsize=(12, 4))
        plt.subplot(131).set_title('Original'), plt.imshow(img, cmap='gray')
        plt.subplot(132).set_title('Binarized'), plt.imshow(bin_img, cmap='gray')
        plt.subplot(133).set_title('Skeleton'), plt.imshow(skeleton, cmap='gray')
        plt.show()

    # Построение графа
    G = nx.Graph()
    y_coords, x_coords = np.where(skeleton == 255)

    if len(x_coords) == 0:
        print(f"Нет узлов для графа: {image_path}")
        return None

    # Добавление узлов с яркостью
    for x, y in zip(x_coords, y_coords):
        G.add_node((x, y), brightness=int(img[y, x]))

    # Добавление ребер между 8-связными соседями
    for node in G.nodes():
        x, y = node
        neighbors = [(x + dx, y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if not (dx == 0 and dy == 0)]
        for n in neighbors:
            if n in G.nodes:
                G.add_edge(node, n)

    print(f"Создан граф для {image_path}: {G.number_of_nodes()} узлов, {G.number_of_edges()} ребер")
    return G


def get_subgraphs(graph, walk_depth=2):
    """Рекурсивное извлечение подграфов с заданной глубиной"""
    subgraphs = []
    for node in graph.nodes():
        subgraph = nx.ego_graph(graph, node, radius=walk_depth)
        if subgraph.number_of_nodes() > 0:
            subgraphs.append(subgraph)
    return subgraphs


def generate_walks(subgraphs):
    """Генерация последовательностей узлов для всех подграфов"""
    walks = []
    for sg in subgraphs:
        if sg.number_of_nodes() == 0:
            continue
        try:
            start_node = list(sg.nodes())[0]
            walk = list(nx.dfs_preorder_nodes(sg, start_node))
            walks.append([str(node) for node in walk])
        except:
            continue
    return walks


def create_graph_embeddings(graphs, w2v_model, embed_size=128):
    """Создание векторных представлений для исходных графов"""
    embeddings = []
    for original_graph in graphs:
        nodes = original_graph.nodes()
        node_vectors = []
        for node in nodes:
            node_str = str(node)
            if node_str in w2v_model.wv:
                node_vectors.append(w2v_model.wv[node_str])
        if node_vectors:
            avg_vector = np.mean(node_vectors, axis=0)
        else:
            avg_vector = np.zeros(embed_size)
        embeddings.append(avg_vector)
    return np.array(embeddings)


def build_classifier(input_dim, num_classes):
    """Построение классификатора согласно статье"""
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def main():
    CLASSES = 2
    EMBED_SIZE = 128
    WALK_DEPTH = 2
    EPOCHS = 128
    BATCH_SIZE = 64

    # Загрузка данных и преобразование в графы
    graphs = []
    labels = []
    valid_images = 0
    for class_id, class_dir in enumerate(["bad_red_cut", "good_red_cut"]):
        if not os.path.exists(class_dir):
            raise FileNotFoundError(f"Директория {class_dir} не найдена")

        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            G = image_to_graph(img_path)
            if G and G.number_of_nodes() > 10:
                graphs.append(G)
                labels.append(class_id)
                valid_images += 1

    print(f"Успешно обработано изображений: {valid_images}")
    if valid_images == 0:
        raise ValueError("Нет подходящих изображений для обработки")

    # Преобразование меток в numpy массив
    labels = np.array(labels)


    # Извлечение подграфов
    subgraphs = []
    for g in graphs:
        subgraphs.extend(get_subgraphs(g, WALK_DEPTH))

    # Генерация последовательностей узлов
    walks = generate_walks(subgraphs)
    if not walks:
        raise ValueError("Не удалось сгенерировать последовательности для обучения.")

    # Обучение Word2Vec
    w2v_model = Word2Vec(
        walks,
        vector_size=EMBED_SIZE,
        window=5,
        min_count=1,
        negative=15,
        epochs=10,
        sg=1
    )

    # Создание векторных представлений

    embeddings = create_graph_embeddings(graphs, w2v_model, EMBED_SIZE)

    # Проверка формы и типа эмбеддингов
    print(f"Форма эмбеддингов: {embeddings.shape}")
    print(f"Тип эмбеддингов: {embeddings.dtype}")

    # Преобразование к float32 для TensorFlow
    embeddings = embeddings.astype(np.float32)

    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.15, stratify=labels
    )

    # Проверка типов после разделения
    print(f"Тип X_train: {type(X_train)}, Форма: {X_train.shape}")
    print(f"Тип y_train: {type(y_train)}, Форма: {y_train.shape}")

    # Обучение классификатора
    model = build_classifier(EMBED_SIZE, CLASSES)
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    # Оценка и визуализация
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nТочность на тестовых данных: {test_acc:.2%}")

    # Визуализация кривых обучения
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Обучение')
    plt.plot(history.history['val_accuracy'], label='Валидация')
    plt.title('Кривые точности')
    plt.xlabel('Эпохи')
    plt.ylabel('Точность')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Обучение')
    plt.plot(history.history['val_loss'], label='Валидация')
    plt.title('Кривые потерь')
    plt.xlabel('Эпохи')
    plt.ylabel('Потери')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Матрица ошибок
    y_pred = model.predict(X_test).argmax(axis=1)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['0nM', '40nM'],
                yticklabels=['0nM', '40nM'])
    plt.title('Матрица ошибок')
    plt.xlabel('Предсказание')
    plt.ylabel('Истина')
    plt.show()

    # Визуализация PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(embeddings)

    # Создаем фигуру
    plt.figure(figsize=(10, 7))

    # Создаем scatter plot
    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=labels,
        cmap='viridis',
        alpha=0.7,
        s=50,  # Размер точек
        edgecolor='w',  # Белая граница
        linewidth=0.5
    )

    # Добавляем подписи
    plt.title('2D проекция векторных представлений графов', fontsize=14)
    plt.xlabel(f'Главная компонента 1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
    plt.ylabel(f'Главная компонента 2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)

    # Добавляем цветовую легенду
    cbar = plt.colorbar(scatter)
    cbar.set_label('Классы', fontsize=12)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Контроль', 'Остеогенная дифференцировка'])

    # Добавляем сетку для лучшей читаемости
    plt.grid(alpha=0.3, linestyle='--')

    # Оптимизируем расположение элементов
    plt.tight_layout()

    # Показываем график
    plt.show()


if __name__ == "__main__":
    main()