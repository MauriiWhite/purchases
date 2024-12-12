import os
import warnings

from typing import List, Optional, Dict, Union


class Config:
    def __init__(
        self,
        file_path: str = "data/purchases.xlsx",
        features: Optional[List[str]] = None,
        cpu_cores: Optional[int] = 2,
    ):
        if features is None:
            features = [
                "CUPO MÁXIMO",
                "PORCENTAJE DE USO DEL CUPO",
                "VECES QUE COMPRA EN PROMEDIO AL AÑO",
                "UNIDADES COMPRADAS DEL PRODUCTO A",
                "UNIDADES COMPRADAS DEL PRODUCTO B",
                "CANTIDAD HISTORICA DE ATRASOS EN PAGOS",
            ]
        self._file_path = file_path
        self._features = features
        self._cpu_cores = cpu_cores

        self._set_environment()

    def _set_environment(self) -> None:
        os.environ["LOKY_MAX_CPU_COUNT"] = str(self.cpu_cores)
        warnings.filterwarnings("ignore", category=UserWarning)

    def update(
        self,
        file_path: Optional[str] = None,
        features: Optional[List[int]] = None,
        cpu_cores: Optional[int] = None,
    ) -> None:
        if file_path:
            self.file_path = file_path
        if features:
            self.features = features
        if cpu_cores:
            self.cpu_cores = cpu_cores
        self._set_environment()

    def get_config(self) -> Dict[str, Union[str, List[str], int]]:
        return {
            "file_path": self.file_path,
            "features": self.features,
            "cpu_cores": self.cpu_cores,
        }

    @property
    def file_path(self) -> str:
        return self._file_path

    @property
    def features(self) -> List[str]:
        return self._features

    @property
    def cpu_cores(self) -> int:
        return self._cpu_cores
