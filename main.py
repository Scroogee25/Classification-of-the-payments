import pandas as pd
import numpy as np
from gensim.models import FastText
import re
from sklearn.neighbors import KNeighborsClassifier


# Регулярное выражение для токенизации слов
WORD_PATTERN = r'(?u)\b\w+\b|№'


def update_data(data):
    data['Description'] = data['Description'].str.replace(r'[\d\-]+', '', regex=True)
    data['Description'] = data['Description'].str.replace(r'№(\w)', r'№ \1', regex=True)
    data['Description'] = data['Description'].str.lower()
    data = data.drop(columns=['Date'], errors='ignore')
    return data


# Токенизация заголовков
def tokenize_titles(titles):
    reg_exp = re.compile(WORD_PATTERN)
    return [reg_exp.findall(title.lower()) for title in titles]


class FastTextTransformer:
    def __init__(self, ft_model, word_pattern):
        self.ft_model = ft_model
        self.word_pattern = re.compile(word_pattern)

    def transform(self, titles):
        # Создание массива для хранения векторов заголовков
        transformed = np.zeros((len(titles), self.ft_model.wv.vector_size))
        for i, title in enumerate(titles):
            # Токенизация заголовка
            tokens = self.word_pattern.findall(title.lower())
            # Получение векторов слов для токенов
            word_vectors = [self.ft_model.wv[token] for token in tokens if token in self.ft_model.wv.key_to_index]

            # Если список векторов не пуст, вычисляем средний вектор
            if word_vectors:
                title_vector = np.mean(word_vectors, axis=0)
                # Проверка на NaN и замена на нули при необходимости
                if np.isnan(title_vector).any():
                    title_vector = np.zeros(self.ft_model.wv.vector_size)

                transformed[i] = title_vector
            else:
                # Если нет известных слов, используем вектор из нулей
                transformed[i] = np.zeros(self.ft_model.wv.vector_size)
        return transformed


def main():
    # Путь к файлам с обучающими и тестовыми данными
    train_path = 'train.tsv'
    test_path = 'payments_main.tsv'
    columns = ['ID', 'Date', 'Amount', 'Description', 'Service']
    
    # Загрузка данных
    train = pd.read_csv(train_path, sep='\t', header=None, names=columns)
    test = pd.read_csv(test_path, sep='\t', header=None, names=columns[:-1])
    initial_test = test.copy()
    
    # Обновление данных
    train = update_data(train)
    test = update_data(test)
    
    # Токенизация описаний
    train_sentences = tokenize_titles(train['Description'])
    unlabeled_sentences = tokenize_titles(test['Description'])
    
    # Создание и обучение модели FastText
    ft_model = FastText(vector_size=128, window=5, min_count=1, sg=1)
    ft_model.build_vocab(train_sentences + unlabeled_sentences)

    ft_model.train(
        train_sentences + unlabeled_sentences,
        total_examples=ft_model.corpus_count,
        epochs=5,
        compute_loss=True
    )
    
    # Преобразование заголовков в векторы
    ft_transformer = FastTextTransformer(ft_model=ft_model, word_pattern=WORD_PATTERN)
    train_embd = ft_transformer.transform(train['Description'].values)
    test_embd = ft_transformer.transform(test['Description'].values)
    
    # Создание отображений классов и меток
    class_to_label = dict()
    label_to_class = dict()
    for i, class_name in enumerate(train['Service'].unique()):
        class_to_label[class_name] = i
        label_to_class[i] = class_name
        
    # Преобразование меток в числовой формат
    train_labels = np.array([class_to_label[x] for x in train['Service'].values])

    # Обучение классификатора KNN
    knn = KNeighborsClassifier(n_neighbors=7, metric='cosine')
    knn.fit(train_embd, train_labels)
    
    # Прогнозирование меток для тестового набора
    predicted_labels = knn.predict(test_embd)
    
    # Объединение исходного тестового набора с предсказанными метками
    test_with_labels = pd.concat([initial_test, 
        pd.Series(list(label_to_class[x] for x in predicted_labels), name='Class')], axis=1)
    
    # Создание DataFrame с ID и предсказанными метками
    test_id_with_labels = pd.concat([initial_test['ID'], 
        pd.Series(list(label_to_class[x] for x in predicted_labels), name='Class')], axis=1)
    
    # Сохранение предсказаний в файл
    test_id_with_labels.to_csv('predictions.tsv', index=False, header=False)

# Запуск основной функции
if __name__ == "__main__":
    main()