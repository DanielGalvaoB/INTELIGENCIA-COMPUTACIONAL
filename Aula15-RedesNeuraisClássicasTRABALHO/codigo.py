# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Carregamento dos dados
iris = load_iris()
# Usando atributos 2 e 3 (Petal length e Petal width) para visualização 2D
X = iris.data[:, [2, 3]] 
y = iris.target

# 2. Definição dos modelos
# O SGDClassifier com loss='squared_error' emula o comportamento do ADALINE
modelos = {
    "Perceptron": Perceptron(max_iter=1000, eta0=0.01, random_state=42),
    "ADALINE": SGDClassifier(loss='squared_error', max_iter=1000, learning_rate='constant', eta0=0.01, random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(10, 10), activation='relu', max_iter=2000, random_state=42)
}

# 3. Função do Experimento (Repetições)
def run_experiment(modelo_nome, modelo, n_runs=30, test_size=0.2):
    accs = []
    conf_total = np.zeros((3, 3), dtype=int)

    for _ in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        accs.append(accuracy_score(y_test, y_pred))
        conf_total += confusion_matrix(y_test, y_pred)

    return np.array(accs), conf_total

# 4. Execução e Impressão das Métricas
print("="*50)
print("MÉTRICAS DOS MODELOS (30 EXECUÇÕES)")
print("="*50)

for nome, modelo in modelos.items():
    accs, conf_matrix = run_experiment(nome, modelo)
    print(f"\n--- {nome} ---")
    print(f"Acurácia média: {accs.mean():.4f}")
    print(f"Desvio padrão:  {accs.std():.4f}")
    print("Matriz de confusão acumulada:")
    print(conf_matrix)

# 5. Visualização das Fronteiras de Decisão
# Treinamento de uma rodada única para gerar os gráficos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Preparando a malha de pontos
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

# Criando a figura com 3 subplots lado a lado
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
cores = ['red', 'green', 'blue']

for ax, (nome, modelo) in zip(axes, modelos.items()):
    # Treina o modelo
    modelo.fit(X_train_scaled, y_train)
    
    # Prevê os valores da malha
    Z = modelo.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plota a região de decisão
    ax.contourf(xx, yy, Z, alpha=0.3)
    
    # Plota os pontos reais
    for class_value, color, label in zip([0, 1, 2], cores, iris.target_names):
        ax.scatter(
            X_train_scaled[y_train == class_value, 0],
            X_train_scaled[y_train == class_value, 1],
            c=color, label=label, edgecolor='k'
        )
        
    ax.set_title(f"{nome} - Fronteiras")
    ax.set_xlabel("Petal length (normalizado)")
    ax.set_ylabel("Petal width (normalizado)")
    ax.legend()

plt.tight_layout()
plt.savefig('fronteiras_redes_neurais.png', dpi=300)
plt.show()