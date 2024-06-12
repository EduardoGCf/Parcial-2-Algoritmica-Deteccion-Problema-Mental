import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import joblib  # Para guardar y cargar el modelo
from sklearn.impute import SimpleImputer


# Cargar los datos desde el archivo CSV
df = pd.read_csv('Deepression.csv')
# Crear una instancia del modelo KNN
knn_model = KNeighborsClassifier(n_neighbors=5)  # Puedes ajustar el número de vecinos aquí

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

# Imputación de valores faltantes
imputer = SimpleImputer(strategy='mean')  # Estrategia de imputación: rellenar con la media
X_test_imputed = imputer.fit_transform(X_test)
X_test_imputed = imputer.transform(X_test)
y_pred = knn_model.predict(X_test_imputed)


# Entrenar el modelo KNN con los datos imputados
knn_model.fit(X_train_imputed, y_train)

# Guardar el modelo y el scaler
joblib.dump(knn_model, 'depression_knn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("\nModelo y scaler guardados exitosamente.")

# Hacer predicciones en el conjunto de prueba
y_pred = knn_model.predict(X_test)

# Convertir las etiquetas a cadenas de texto
label_encoder_classes_str = [str(label) for label in label_encoder.classes_]

# Evaluar el rendimiento del modelo
report = classification_report(y_test, y_pred, labels=label_encoder.transform(label_encoder.classes_), target_names=label_encoder_classes_str)

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
#tabla de nombre de las caracteristicas
#Sleep: Frequency of sleep disturbances (1: Never, 2: Always, 3: Often, 4: Rarely, 5: Sometimes, 6: Not at all).
#Appetite: Changes in appetite (1: Never, 2: Always, 3: Often, 4: Rarely, 5: Sometimes, 6: Not at all).
#Interest: Loss of interest in activities (1: Never, 2: Always, 3: Often, 4: Rarely, 5: Sometimes, 6: Not at all).
#Fatigue: Feelings of fatigue or low energy (1: Never, 2: Always, 3: Often, 4: Rarely, 5: Sometimes, 6: Not at all).
#Worthlessness: Feelings of worthlessness or excessive guilt (1: Never, 2: Always, 3: Often, 4: Rarely, 5: Sometimes, 6: Not at all).
#Concentration: Difficulty concentrating (1: Never, 2: Always, 3: Often, 4: Rarely, 5: Sometimes, 6: Not at all).
#Agitation: Physical agitation (1: Never, 2: Always, 3: Often, 4: Rarely, 5: Sometimes, 6: Not at all).
#Suicidal Ideation: Thoughts of self-harm or suicide (1: Never, 2: Always, 3: Often, 4: Rarely, 5: Sometimes, 6: Not at all).
#Sleep Disturbance: Issues with sleeping (1: Never, 2: Always, 3: Often, 4: Rarely, 5: Sometimes, 6: Not at all).
#Aggression: Feelings of aggression (1: Never, 2: Always, 3: Often, 4: Rarely, 5: Sometimes, 6: Not at all).
#Panic Attacks: Experiencing panic attacks (1: Never, 2: Always, 3: Often, 4: Rarely, 5: Sometimes, 6: Not at all).
#Hopelessness: Feelings of hopelessness (1: Never, 2: Always, 3: Often, 4: Rarely, 5: Sometimes, 6: Not at all).
#Restlessness: Feelings of restlessness (1: Never, 2: Always, 3: Often, 4: Rarely, 5: Sometimes, 6: Not at all).
#Low Energy: Lack of energy (1: Never, 2: Always, 3: Often, 4: Rarely, 5: Sometimes, 6: Not at all).
#Depression State: Overall depression state (categorized as 'No depression', 'Mild', 'Moderate', 'Severe').
Sleep= 2
Appetite= 3
Interest=3
Fatigue=3
Worthlessness=3
Concentration=3
Agitation=3
Suicidal_Ideation=3
Sleep_Disturbance=3
Aggression=3
Panic_Attacks=3
Hopelessness=3
Restlessness=3
Low_Energy=3
nuevas_caracteristicas = [Sleep, Appetite, Interest, Fatigue, Worthlessness, Concentration, Agitation, Suicidal_Ideation, Sleep_Disturbance, Aggression, Panic_Attacks, Hopelessness, Restlessness, Low_Energy]  # Ejemplo de características del usuario
diagnostico = diagnosticar_usuario(nuevas_caracteristicas)
print("\nDiagnóstico del usuario:", diagnostico)
