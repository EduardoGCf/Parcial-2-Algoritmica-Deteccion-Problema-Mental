import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
import joblib

# Cargar los datos desde el archivo CSV
df = pd.read_csv('Deepression.csv')

# Eliminar columnas 'Number' y 'Depression State'
df = df.rename(columns=lambda x: x.strip())  # Corregir el nombre de la columna eliminando el espacio al final
X = df.drop(['Number', 'Depression State'], axis=1)  # Características
y = df['Depression State']  # Etiquetas

# Codificar la columna de etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Codificar las etiquetas de texto a números

# Escalar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Imputación de valores faltantes en el conjunto de entrenamiento
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)

# Ajustar el modelo a los datos de entrenamiento
knn_model = KNeighborsClassifier(n_neighbors=10)
knn_model.fit(X_train_imputed, y_train)

# Imputación de valores faltantes en el conjunto de prueba
X_test_imputed = imputer.transform(X_test)

# Hacer predicciones en el conjunto de prueba
y_pred = knn_model.predict(X_test_imputed)

# Guardar el modelo y el scaler
joblib.dump(knn_model, 'depression_knn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("\nModelo y scaler guardados exitosamente.")

# Obtener las clases únicas del conjunto de datos
unique_classes = label_encoder.classes_

# Convertir los nombres de las clases a cadenas
unique_classes_str = [str(cls) for cls in unique_classes]

# Evaluar el rendimiento del modelo
report = classification_report(y_test, y_pred, labels=range(len(unique_classes)), target_names=unique_classes_str)
print("\nReporte de clasificación:")
print(report)


# Función para diagnosticar a nuevos usuarios
def diagnosticar_usuario_knn(nuevas_caracteristicas):
    # Cargar el modelo entrenado, el scaler y el label encoder
    modelo = joblib.load('depression_knn_model.pkl')
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
Sleep = 1  # Rarely
Appetite = 3  # Often
Interest = 2  # Always
Fatigue = 4  # Rarely
Worthlessness = 5  # Sometimes
Concentration = 3  # Often
Agitation = 2  # Always
Suicidal_Ideation = 5  # Never
Sleep_Disturbance = 5  # Rarely
Aggression = 1  # Never
Panic_Attacks = 3  # Never
Hopelessness = 4  # Rarely
Restlessness = 5  # Sometimes
Low_Energy = 4  # Rarely
nuevas_caracteristicas = [Sleep, Appetite, Interest, Fatigue, Worthlessness, Concentration, Agitation, Suicidal_Ideation, Sleep_Disturbance, Aggression, Panic_Attacks, Hopelessness, Restlessness, Low_Energy]  # Ejemplo de características del usuario
diagnostico = diagnosticar_usuario_knn(nuevas_caracteristicas)
print("\nDiagnóstico del usuario:", diagnostico)
