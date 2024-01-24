import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from joblib import dump, load
import time

class CodeLanguagePredictor:
    def __init__(self, data_file, model_file, vectorizer_file):
        self.data_file = data_file
        self.model_file = model_file
        self.vectorizer_file = vectorizer_file

    def train_model_and_save(self):
        data = pd.read_csv(self.data_file)

        vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
        X_train_transformed = vectorizer.fit_transform(data['code'].fillna(' '))

        start_time = time.time()

        model = LinearSVC(dual=False)
        model.fit(X_train_transformed, data['lang'])

        dump(model, self.model_file)
        dump(vectorizer, self.vectorizer_file)

        training_accuracy = model.score(X_train_transformed, data['lang'])
        print(f"Точность модели: {training_accuracy:.2f}")
        end_time = time.time()

        training_time = end_time - start_time
        print(f"Время загрузки: {training_time:.2f} секунд")

    def predict_language(self, code_snippet):
        model = load(self.model_file)
        vectorizer = load(self.vectorizer_file)

        code_snippet_transformed = vectorizer.transform([code_snippet])

        predicted_language = model.predict(code_snippet_transformed)
        return predicted_language[0]

    @staticmethod
    def contains_only_digits(input_str):
        return all(char.isdigit() for char in input_str)

    def run_prediction_loop(self):
        self.train_model_and_save()

        while True:
            code_snippet = input('Введите фрагмент кода (напишите выход чтобы закрыть программу): ')
            if code_snippet.lower() == 'выход':
                break
            elif self.contains_only_digits(code_snippet):
                print("Пожалуйста, введите фрагмент кода, который содержит не только цифры.")
            else:
                predicted_language = self.predict_language(code_snippet)
                print(f"Прогнозируемый язык программирования: {predicted_language}")

def main():
    predictor = CodeLanguagePredictor('dataset.csv', 'svm_model.joblib', 'tfidf_vectorizer.joblib')
    predictor.run_prediction_loop()

if __name__ == "__main__":
    main()