from typing import List

from numpy import ndarray, array
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from settings.config import Config
from utils import DataProcessor, Visualizer, Segmenter, ModelTrainer, PredictionEngine


class CustomerModel:
    def __init__(
        self,
        file_path: str,
        features: List[str],
        random_state: int = 42,
        n_clusters: int = 3,
    ) -> None:
        self.file_path = file_path
        self.features = features
        self.random_state = random_state
        self.n_clusters = n_clusters

        self.data_processor = DataProcessor(self.file_path, self.features)
        self.segmenter = Segmenter(
            n_clusters=self.n_clusters, random_state=self.random_state
        )
        self.visualizer = Visualizer()
        self.label_encoder = LabelEncoder()
        self.model_trainer = ModelTrainer(random_state=self.random_state)
        self.prediction_engine = None

    def load_and_preprocess_data(self) -> None:
        self.data_processor.load_data()
        self.df_cleaned, self.scaled_data, self.scaler = (
            self.data_processor.preprocess_data()
        )

    def segment_data(self) -> None:
        self.df_cleaned["Segment"] = self.segmenter.segment(self.scaled_data)

    def visualize_segments(self) -> None:
        self.visualizer.plot_segments(self.df_cleaned["Segment"])

    def encode_labels(self) -> None:
        self.df_cleaned["CompraPromo"] = self.label_encoder.fit_transform(
            self.df_cleaned["COMPRA PRODUCTO EN PROMOCIÓN"]
        )

    def train_and_predict(self) -> None:
        prediction_features = self.features + ["Segment"]
        X = self.df_cleaned[prediction_features]
        y = self.df_cleaned["CompraPromo"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state
        )

        self.model_trainer.train(X_train, y_train)
        y_pred = self.model_trainer.predict(X_test)
        self.visualizer.plot_confusion_matrix(
            self.model_trainer.model, X_test, y_test, self.label_encoder
        )
        self.model_trainer.evaluate(y_test, y_pred, self.label_encoder)

        self.prediction_engine = PredictionEngine(
            self.segmenter,
            self.model_trainer,
            self.visualizer,
            self.scaler,
            self.label_encoder,
        )

    def predict_for_new_customer(self, customer_data: ndarray) -> None:
        if not self.prediction_engine:
            raise RuntimeError("The model has not been trained yet")

        self.prediction_engine.predict_customer(
            customer_data, self.segmenter.kmeans, self.model_trainer.model
        )


if __name__ == "__main__":
    config = Config()

    customer = CustomerModel(config.file_path, config.features)
    customer.load_and_preprocess_data()
    customer.segment_data()
    customer.visualize_segments()
    customer.encode_labels()
    customer.train_and_predict()

    # Cliente que compra en promoción
    new_customer_1 = array(
        [
            5000,  # CUPO MÁXIMO
            75,  # PORCENTAJE DE USO DEL CUPO
            12,  # VECES QUE COMPRA EN PROMEDIO AL AÑO
            30,  # UNIDADES COMPRADAS DEL PRODUCTO A
            10,  # UNIDADES COMPRADAS DEL PRODUCTO B
            2,  # CANTIDAD HISTÓRICA DE ATRASOS EN PAGOS
        ]
    )
    customer.predict_for_new_customer(new_customer_1)

    # Cliente que no compra en promoción
    new_customer_2 = array([0, 0, 0, 0, 0, 0])
    customer.predict_for_new_customer(new_customer_2)
