#Ejecuta la celda a continuación para configurar tu entorno de codificación desde donde quedó el ejercicio anterior.

#Código inicial para configurar el entorno

#add Codeadd Markdown
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer

# Cargar los datos desde el archivo CSV
df = pd.read_csv('Deepression.csv')

# Eliminar columnas 'Number' y 'Depression State'
df = df.rename(columns=lambda x: x.strip())  # Corregir el nombre de la columna eliminando el espacio al final
X = df.drop(['Number', 'Depression State'], axis=1)  # Características
y = df['Depression State']  # Etiquetas​
# Codificar la columna de etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Codificar las etiquetas de texto a números

# Escalar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)​
print("Setup Complete")

#add Codeadd Markdown
#Paso 1: Dividir tus datos Usa la función train_test_split para dividir tus datos.

#Dale el argumento random_state=42 para que las funciones de verificación sepan qué esperar al verificar tu código.

#Recuerda, tus características están cargadas en el DataFrame X_scaled y tu objetivo está cargado en y_encoded.

#add Codeadd Markdown
# Importa la función train_test_split y descomenta
from sklearn.model_selection import train_test_split

# Completa y descomenta
X_train, X_test, y_train, y_test = train_test_split(---, ---, test_size=0.2, random_state=42)

# Verifica tu respuesta
# step_1.check()