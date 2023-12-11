import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

class MathScorePredictorApp(QWidget):
    def __init__(self):
        super().__init__()
        filename =  "./data/exams.csv"

        self.encoders = {}
        self.data = pd.read_csv(filename)
        self.data = self.data.dropna()

        target = self.data['math score']
        # features = data[list(expected_columns)]
        features = self.data.drop("math score", axis=1)

        features = self.fit_transform(features)

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

        self.best_model = RandomForestRegressor(max_depth=8, max_features='sqrt', n_estimators=200)
        self.best_model.fit(X_train, y_train)

        self.init_ui()

    def fit_transform(self, df):
        for column in df.columns:
            if df[column].dtype == 'object':
                encoder = LabelEncoder()
                df[column] = encoder.fit_transform(df[column])
                self.encoders[column] = encoder
        return df

    def transform(self, df):
        for column, encoder in self.encoders.items():
            if df[column].dtype == 'object':
                df[column] = encoder.transform(df[column])
                print(f"Column: {column}, Classes: {encoder.classes_}")
        return df

    def init_ui(self):
        self.setWindowTitle('Math Score Predictor')

        self.gender_label = QLabel('Gender (male/female):')
        self.gender_input = QLineEdit()
        self.gender_input.setText('female')

        self.ethnicity_label = QLabel('Race/Ethnicity (group A, B, C...):')
        self.ethnicity_input = QLineEdit()
        self.ethnicity_input.setText('group A')

        self.education_label = QLabel('Parental Level of Education (some college, high school...):')
        self.education_input = QLineEdit()
        self.education_input.setText('some college')

        self.lunch_label = QLabel('Lunch (standard / free/reduced):')
        self.lunch_input = QLineEdit()
        self.lunch_input.setText('free/reduced')

        self.prep_course_label = QLabel('Test Preparation Course (completed/none):')
        self.prep_course_input = QLineEdit()
        self.prep_course_input.setText('completed')

        self.reading_score_label = QLabel('Reading score:')
        self.reading_score_input = QLineEdit()
        self.reading_score_input.setText('80')

        self.writing_score_label = QLabel('Writing score:')
        self.writing_score_input = QLineEdit()
        self.writing_score_input.setText('83')

        self.predict_button = QPushButton('Predict Math Score')
        self.predict_button.clicked.connect(self.predict_math_score)

        self.result = QLabel('')

        layout = QVBoxLayout()
        layout.addWidget(self.gender_label)
        layout.addWidget(self.gender_input)
        layout.addWidget(self.ethnicity_label)
        layout.addWidget(self.ethnicity_input)
        layout.addWidget(self.education_label)
        layout.addWidget(self.education_input)
        layout.addWidget(self.lunch_label)
        layout.addWidget(self.lunch_input)
        layout.addWidget(self.prep_course_label)
        layout.addWidget(self.prep_course_input)
        layout.addWidget(self.writing_score_label)
        layout.addWidget(self.writing_score_input)
        layout.addWidget(self.reading_score_label)
        layout.addWidget(self.reading_score_input)
        layout.addWidget(self.predict_button)
        layout.addWidget(self.result)

        self.setLayout(layout)

    def predict_math_score(self):
        gender = self.gender_input.text()
        ethnicity = self.ethnicity_input.text()
        education = self.education_input.text()
        lunch = self.lunch_input.text()
        prep_course = self.prep_course_input.text()
        reading_score = int(self.reading_score_input.text())
        writing_score = int(self.writing_score_input.text())

        input_data = pd.DataFrame({'gender': [gender],
                                   'race/ethnicity': [ethnicity],
                                   'parental level of education': [education],
                                   'lunch': [lunch],
                                   'test preparation course': [prep_course],
                                   'reading score': [writing_score],
                                   'writing score': [reading_score]
                                   })
        input_data = self.transform(input_data)

        math_score_prediction = self.best_model.predict(input_data)

        self.result.setText(f'Predicted Math Score: {np.round(self.best_model.predict(input_data)[0])}')

        print(f'Predicted Math Score: {math_score_prediction}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MathScorePredictorApp()
    window.show()
    sys.exit(app.exec_())