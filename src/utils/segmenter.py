from numpy import ndarray

from sklearn.cluster import KMeans


class Segmenter:
    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
        )

    def segment(self, data: ndarray[float]) -> ndarray:
        return self.kmeans.fit_predict(data)

