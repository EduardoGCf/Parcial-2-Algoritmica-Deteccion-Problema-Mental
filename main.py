import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib  # Para guardar y cargar el modelo

# Cargar los datos desde el archivo CSV
df = pd.read_csv('Deepression.csv')

# Mostrar las primeras filas del DataFrame
print("Primeras filas del DataFrame:")
print(df.head())

# Mostrar información resumida del DataFrame
print("\nDescripción del DataFrame:")
print(df.describe(include='all'))

# Imprimir los nombres de las columnas para diagnóstico
print("\nNombres de las columnas del DataFrame:")
print(df.columns)

# Eliminar columnas 'Number' y 'Depression State'
try:
    # Corregir el nombre de la columna eliminando el espacio al final
    df = df.rename(columns=lambda x: x.strip())
    
    # Eliminar columnas 'Number' y 'Depression State'
    X = df.drop(['Number', 'Depression State'], axis=1)  # Características
    y = df['Depression State']  # Etiquetas
    
    print("\nColumnas 'Number' y 'Depression State' eliminadas con éxito.")
except KeyError as e:
    print(f"\nError al intentar eliminar columnas: {e}")

# Codificar la columna de etiquetas antes de dividir los datos
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['Depression State'])  # Codificar las etiquetas de texto a números

# Escalar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Entrenar el modelo utilizando las etiquetas codificadas
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Guardar el modelo y el scaler
joblib.dump(model, 'depression_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("\nModelo y scaler guardados exitosamente.")

# Hacer predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Convertir las etiquetas a cadenas de texto
label_encoder_classes_str = [str(label) for label in label_encoder.classes_]

# Evaluar el rendimiento del modelo
report = classification_report(y_test, y_pred, labels=label_encoder.transform(label_encoder.classes_), target_names=label_encoder_classes_str)

print("\nReporte de clasificación:")
print(report)



# Función para diagnosticar a nuevos usuarios
def diagnosticar_usuario(nuevas_caracteristicas):
    # Cargar el modelo entrenado, el scaler y el label encoder
    modelo = joblib.load('depression_model.pkl')
    escalador = joblib.load('scaler.pkl')
    codificador = joblib.load('label_encoder.pkl')
    
    # Preprocesar las nuevas características
    nuevas_caracteristicas_esc = escalador.transform([nuevas_caracteristicas])
    
    # Hacer una predicción
    prediccion = modelo.predict(nuevas_caracteristicas_esc)
    
    # Decodificar la etiqueta
    estado_depresion = codificador.inverse_transform(prediccion)
    
    return estado_depresion[0]

# Ejemplo de uso:
#tabla de nombre de las caracteristicas
nuevas_caracteristicas = [2, 3, 4, 5, 1, 3, 4, 5, 2, 1, 3, 4, 2, 5]  # Ejemplo de características del usuario
diagnostico = diagnosticar_usuario(nuevas_caracteristicas)
print("\nDiagnóstico del usuario:", diagnostico)
