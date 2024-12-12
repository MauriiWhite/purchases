from numpy import ndarray
from pandas.core.frame import DataFrame
from pandas.core.series import Series

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


class ModelTrainer:
    def __init__(self, random_state: int = 42) -> None:
        self.model = RandomForestClassifier(random_state=random_state)

    def train(self, X_train: DataFrame, y_train: Series) -> None:
        self.model.fit(X_train, y_train)

    def predict(self, X_test: DataFrame) -> ndarray[int]:
        return self.model.predict(X_test)

    def evaluate(
        self, y_test: Series, y_pred: ndarray, label_encoder: LabelEncoder
    ) -> str:
        report = classification_report(
            y_test, y_pred, target_names=label_encoder.classes_
        )
        print(report)
        return report
