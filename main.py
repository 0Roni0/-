from PyQt5 import QtCore, QtGui, QtWidgets
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from joblib import dump, load
import time
import sys

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(535, 197)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 128, 301, 21))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.predicted_language = QtWidgets.QLabel(self.centralwidget)
        self.predicted_language.setGeometry(QtCore.QRect(326, 130, 181, 21))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.predicted_language.setFont(font)
        self.predicted_language.setText("")
        self.predicted_language.setObjectName("predicted_language")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(20, 60, 491, 20))
        self.lineEdit.setObjectName("lineEdit")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(20, 20, 291, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(384, 90, 131, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Распознование языка"))
        self.label.setText(_translate("MainWindow", "Прогназируемый язык программирования:"))
        self.label_2.setText(_translate("MainWindow", "Введите фрагмент кода:"))
        self.pushButton.setText(_translate("MainWindow", "Распознать"))


class CodeLanguagePredictor:
    def __init__(self, data_file, model_file, vectorizer_file):
        self.data_file = data_file
        self.model_file = model_file
        self.vectorizer_file = vectorizer_file
        self.train_model_and_save()

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
class CodeLanguagePredictorApp(Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self, data_file, model_file, vectorizer_file):
        super().__init__()
        self.setupUi(self)
        self.predictor = CodeLanguagePredictor(data_file, model_file, vectorizer_file)
        self.pushButton.clicked.connect(self.on_predict_button_clicked)

    def on_predict_button_clicked(self):
        code_snippet = self.lineEdit.text()
        predicted_language = self.predictor.predict_language(code_snippet)
        self.predicted_language.setText(predicted_language)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    predictor_app = CodeLanguagePredictorApp('dataset.csv', 'svm_model.joblib', 'tfidf_vectorizer.joblib')
    predictor_app.show()
    sys.exit(app.exec_())