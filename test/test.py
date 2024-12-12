import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def main():
    df = pd.read_excel("data/purchases.xlsx")
    df.rename(
        columns={
            x: x.lower().replace(" ", "_").replace("á", "a").replace("ó", "o")
            for x in df.columns
        },
        inplace=True,
    )

    df["estado_civil"] = df["estado_civil"].fillna("desconocido").str.upper()
    df["actividad"] = df["actividad"].fillna("desconocido").str.upper()

    df = df[df["cupo_maximo"] < 9_000_000]

    encoder = LabelEncoder()
    df["sexo"] = encoder.fit_transform(df["sexo"])
    df["compra_producto_en_promocion"] = encoder.fit_transform(
        df["compra_producto_en_promocion"]
    )

    onehot_encoder = onehot_encoder = OneHotEncoder(drop="first", sparse_output=False)
    categorical_columns = [
        "estado_civil",
        "actividad",
        "rango_etario",
        "nivel_educacional",
    ]

    df_encoded = onehot_encoder.fit_transform(df[categorical_columns])
    df_encoded = pd.DataFrame(
        df_encoded, columns=onehot_encoder.get_feature_names_out(categorical_columns)
    )
    df = df.join(df_encoded)

    scaler = StandardScaler()
    df[
        [
            "cupo_maximo",
            "porcentaje_de_uso_del_cupo",
            "veces_que_compra_en_promedio_al_año",
        ]
    ] = scaler.fit_transform(
        df[
            [
                "cupo_maximo",
                "porcentaje_de_uso_del_cupo",
                "veces_que_compra_en_promedio_al_año",
            ]
        ]
    )

    df.rename(
        columns={
            x: x.lower().replace(" ", "_").replace(".", "_").replace("__", "_")
            for x in df.columns
        },
        inplace=True,
    )

    df_numeric = df.select_dtypes(include=[np.number])
    corr = df_numeric.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
    plt.show()


if __name__ == "__main__":
    main()
