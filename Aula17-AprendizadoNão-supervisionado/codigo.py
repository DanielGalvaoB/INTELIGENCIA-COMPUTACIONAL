import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.stats import mode

# ==========================================
# 1. PREPARAÇÃO DOS DADOS
# ==========================================
print("--- ETAPA 1: Preparação dos Dados ---")

# Carregar o dataset Iris
iris = load_iris()
X = iris.data       # Apenas os 4 atributos: comprimento/largura de sépala e pétala
y_true = iris.target # Guardando os rótulos originais em segredo para a etapa final
feature_names = iris.feature_names

# Criar um DataFrame sem os rótulos de classe para análise
df = pd.DataFrame(X, columns=feature_names)

print(f"Quantidade de amostras: {df.shape[0]}")
print(f"Quantidade de atributos: {df.shape[1]}")
print("\nEstatísticas básicas dos dados:")
print(df.describe().round(2))
print("-" * 40)

# ==========================================
# 2. APLICAÇÃO DE CLUSTERIZAÇÃO (K-MEANS)
# ==========================================
print("\n--- ETAPA 2: Aplicação do K-means ---")

valores_k = [2, 3, 4]
resultados_kmeans = {}

for k in valores_k:
    # Aplicar K-means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    resultados_kmeans[k] = {"modelo": kmeans, "labels": labels}
    
    print(f"\nResultados para k = {k}:")
    
    # Quantidade de elementos em cada grupo
    unique, counts = np.unique(labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        print(f"  Cluster {cluster_id}: {count} elementos")
        
    # Centroides de cada cluster
    centroides = kmeans.cluster_centers_
    for i, centroide in enumerate(centroides):
        print(f"  Centroide {i}: {centroide.round(2)}")

print("-" * 40)

# ==========================================
# 3. VISUALIZAÇÃO DOS CLUSTERS
# ==========================================
print("\n--- ETAPA 3: Gerando Gráficos ---")
# Vamos plotar o Comprimento da Sépala vs Comprimento da Pétala
# (Atributos nas colunas 0 e 2)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, k in enumerate(valores_k):
    modelo = resultados_kmeans[k]["modelo"]
    labels = resultados_kmeans[k]["labels"]
    
    # Scatter plot colorido por cluster
    scatter = axes[i].scatter(X[:, 0], X[:, 2], c=labels, cmap='viridis', edgecolor='k', s=50)
    
    # Marcando os centroides com um 'X' vermelho
    centroides = modelo.cluster_centers_
    axes[i].scatter(centroides[:, 0], centroides[:, 2], c='red', marker='X', s=200, label='Centroides')
    
    axes[i].set_title(f'K-means com k = {k}')
    axes[i].set_xlabel('Comprimento da Sépala (cm)')
    axes[i].set_ylabel('Comprimento da Pétala (cm)')
    axes[i].legend()

plt.tight_layout()
plt.savefig('clusters_kmeans.png', dpi=300)
print("Gráficos salvos com sucesso no arquivo 'clusters_kmeans.png'.")
print("-" * 40)

# ==========================================
# 4. COMPARAÇÃO (ETAPA FINAL)
# ==========================================
print("\n--- ETAPA FINAL: Comparação com as Classes Reais ---")
# O melhor k para o Iris é naturalmente 3, pois sabemos que existem 3 espécies.
# Vamos focar a análise de métricas no k=3.

labels_k3 = resultados_kmeans[3]["labels"]

# Como o K-means não sabe os nomes das classes (ele apenas chama de grupo 0, 1 e 2),
# precisamos mapear os rótulos do K-means para os rótulos reais baseados na moda (valor mais frequente).
labels_mapeados = np.zeros_like(labels_k3)
for i in range(3):
    mask = (labels_k3 == i)
    # A classe real mais frequente dentro deste cluster
    classe_real_predominante = mode(y_true[mask], keepdims=True)[0][0]
    labels_mapeados[mask] = classe_real_predominante

# Calcular Porcentagem de Correspondência (Acurácia)
acuracia = accuracy_score(y_true, labels_mapeados)
print(f"Porcentagem de Correspondência (Acurácia Aproximada): {acuracia * 100:.2f}%\n")

# Calcular Matriz de Confusão Aproximada
cm = confusion_matrix(y_true, labels_mapeados)
print("Matriz de Confusão (Linhas: Real | Colunas: Previsto pelo Cluster):")
cm_df = pd.DataFrame(cm, 
                     index=[f"Real: {name}" for name in iris.target_names],
                     columns=[f"Cluster: {name}" for name in iris.target_names])
print(cm_df)