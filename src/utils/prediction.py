import numpy as np

from numpy import ndarray
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

from .segmenter import Segmenter
from .visualization import Visualizer
from .trainer import ModelTrainer


class PredictionEngine:
    def __init__(
        self,
        segmenter: Segmenter,
        model_trainer: ModelTrainer,
        visualizer: Visualizer,
        scaler: StandardScaler,
        label_encoder: LabelEncoder,
    ):
        self.segmenter = segmenter
        self.model_trainer = model_trainer
        self.visualizer = visualizer
        self.scaler = scaler
        self.label_encoder = label_encoder

    def predict_customer(
        self, new_customer: ndarray[int], kmeans: KMeans, model: RandomForestClassifier
    ) -> ndarray[float]:
        new_customer_scaled = self.scaler.transform(new_customer.reshape(1, -1))
        customer_segment = kmeans.predict(new_customer_scaled)[0]

        print(f"Segmentación del cliente: {customer_segment}")

        new_customer_with_segment = np.hstack(
            [new_customer_scaled, customer_segment.reshape(-1, 1)]
        )

        will_buy = model.predict(new_customer_with_segment)
        prediction = self.label_encoder.inverse_transform(will_buy)[0]

        print(f"Predicción de compra de promoción: {prediction}")
        return new_customer_scaled
