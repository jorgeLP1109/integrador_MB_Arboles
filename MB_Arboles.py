import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


df = pd.read_csv('datos.csv')


print("Columnas disponibles en el DataFrame:")
print(df.columns)


required_columns = ['edad', 'sexo', 'anemico', 'diabetico', 'fumador', 'muerto']
if all(column in df.columns for column in required_columns):
   
    df = df.dropna()

    
    df = pd.get_dummies(df, columns=['sexo'], drop_first=True)

    # Graficar la distribución de clases
    plt.figure(figsize=(8, 6))
    df['muerto'].value_counts().plot(kind='bar', color=['skyblue', 'orange'])
    plt.title('Distribución de Clases')
    plt.xlabel('Muerto')
    plt.ylabel('Frecuencia')
    plt.show()

    # Realizar la partición del conjunto de datos de manera estratificada
    X = df.drop('muerto', axis=1)
    y = df['muerto']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Ajustar un árbol de decisión
    
    tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree_model.fit(X_train, y_train)

    # Calcular el accuracy sobre el conjunto de prueba
    y_pred = tree_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy del árbol de decisión: {accuracy:.4f}')

else:
    print("Faltan columnas requeridas en el DataFrame.")

