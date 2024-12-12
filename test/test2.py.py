# Importar librerías necesarias
# import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE


def main():
    # Leer el archivo de datos
    df = pd.read_excel("data/purchases.xlsx")

    # Limpiar nombres de columnas
    df.rename(
        columns={
            x: x.lower().replace(" ", "_").replace("á", "a").replace("ó", "o")
            for x in df.columns
        },
        inplace=True,
    )

    # Tratar valores faltantes
    df["estado_civil"] = df["estado_civil"].fillna("desconocido").str.upper()
    df["actividad"] = df["actividad"].fillna("desconocido").str.upper()

    # Codificar columnas categóricas usando LabelEncoder
    encoder = LabelEncoder()
    df["sexo"] = encoder.fit_transform(df["sexo"])
    df["compra_producto_en_promocion"] = encoder.fit_transform(
        df["compra_producto_en_promocion"]
    )
    df["estado_civil"] = encoder.fit_transform(df["estado_civil"])
    df["actividad"] = encoder.fit_transform(df["actividad"])
    df["rango_etario"] = encoder.fit_transform(df["rango_etario"])
    df["nivel_educacional"] = encoder.fit_transform(df["nivel_educacional"])
    df["estado_actual"] = encoder.fit_transform(df["estado_actual"])

    # Seleccionar características relevantes
    selected_features = [
        "cantidad_historica_de_atrasos_en_pagos",
        "unidades_compradas_del_producto_a",
        "unidades_compradas_del_producto_b",
        "año_apertura_tarjeta",
        "veces_que_compra_en_promedio_al_año",
        "porcentaje_de_uso_del_cupo",
    ]

    # Variable objetivo
    y = df["compra_producto_en_promocion"]
    # Características
    X = df[selected_features]

    # Dividir los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Balancear las clases utilizando SMOTE
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Crear el modelo Random Forest
    model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)

    # Entrenar el modelo con los datos balanceados
    model.fit(X_train, y_train)

    # Realizar predicciones
    predictions = model.predict(X_test)

    # Evaluar el modelo
    acc = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    class_report = classification_report(y_test, predictions)

    # Mostrar los resultados
    print("Accuracy:", acc)
    print("\nMatriz de Confusión:")
    print(conf_matrix)
    print("\nReporte de Clasificación:")
    print(class_report)

    # Opcional: Visualizar la importancia de características
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame(
        {"Feature": selected_features, "Importance": importances}
    ).sort_values(by="Importance", ascending=False)

    print("\nImportancia de características:")
    print(feature_importance_df)

    # Visualizar la importancia de las características
    sns.barplot(x="Importance", y="Feature", data=feature_importance_df)
    plt.title("Importancia de características")
    plt.show()


if __name__ == "__main__":
    main()
