import seaborn as sns
import matplotlib.pyplot as plt

from pandas.core.series import Series
from pandas.core.frame import DataFrame
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


class Visualizer:
    SEGMENTS = {
        0: "Clientes Frecuentes",
        1: "Clientes Ocasionales",
        2: "Clientes Inactivos",
    }

    def __init__(self) -> None:
        pass

    def plot_segments(self, segments: Series) -> None:
        segment_counts = segments.value_counts()
        segment_counts = segment_counts.rename(index=self.SEGMENTS)

        plt.figure(figsize=(8, 6))
        plt.pie(
            segment_counts,
            labels=segment_counts.index,
            autopct="%1.3f%%",
            startangle=140,
            colors=sns.color_palette("viridis", len(segment_counts)),
        )
        plt.title("Distribución de Segmentos de Clientes")
        plt.show()

    def plot_confusion_matrix(
        self,
        model: RandomForestClassifier,
        X_test: DataFrame,
        y_test: Series,
        label_encoder: LabelEncoder,
    ) -> None:
        ConfusionMatrixDisplay.from_estimator(
            model, X_test, y_test, display_labels=label_encoder.classes_, cmap="Blues"
        )
        plt.title("Confusion Matrix")
        plt.show()
