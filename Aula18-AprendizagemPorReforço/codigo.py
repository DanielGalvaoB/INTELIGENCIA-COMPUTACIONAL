import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.metrics import accuracy_score, confusion_matrix

# ==========================================
# 1. PREPARAÇÃO DOS DADOS E AMBIENTE
# ==========================================

class IrisEnvironment:
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.n_samples = len(features)
        self.current_step = 0

    def reset(self):
        """Reinicia o ambiente para a primeira amostra."""
        self.current_step = 0
        return self._get_state()

    def _get_state(self):
        """Converte o vetor de características em uma tupla para ser usada como estado."""
        return tuple(self.features[self.current_step])

    def step(self, action):
        """Executa a ação, retorna recompensa e próximo estado."""
        correct_label = self.labels[self.current_step]
        
        # Recompensa: +1 se acertar, -1 se errar
        reward = 1.0 if action == correct_label else -1.0
        
        self.current_step += 1
        done = self.current_step >= self.n_samples
        next_state = self._get_state() if not done else None
        
        return next_state, reward, done

# ==========================================
# 2. DEFINIÇÃO DO AGENTE Q-LEARNING
# ==========================================

class QLearningAgent:
    def __init__(self, n_actions=3, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.n_actions = n_actions
        self.alpha = alpha       # Taxa de aprendizado
        self.gamma = gamma       # Fator de desconto
        self.epsilon = epsilon   # Política epsilon-greedy
        self.q_table = {}        # Dicionário para armazenar os valores Q

    def get_q_values(self, state):
        """Retorna os valores Q para um estado. Inicializa com zeros se não existir."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        return self.q_table[state]

    def choose_action(self, state, train=True):
        """Escolhe a ação baseada na política epsilon-greedy."""
        if train and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions) # Exploração
        else:
            return np.argmax(self.get_q_values(state)) # Aproveitamento

    def update_q_value(self, state, action, reward, next_state):
        """Aplica a equação de atualização do Q-learning."""
        current_q = self.get_q_values(state)[action]
        
        if next_state is None:
            max_next_q = 0.0 # Estado terminal
        else:
            max_next_q = np.max(self.get_q_values(next_state))
            
        # Equação do Q-Learning
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q

# ==========================================
# 3. FUNÇÃO DE TREINO E TESTE
# ==========================================

def run_experiment(n_episodes=1000):
    # Carregar dataset Iris
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Dividir 70% treino, 30% teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)
    
    # Normalizar e Discretizar (Fundamental para Q-Learning)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Transforma os números contínuos em 5 categorias (bins) para criar estados finitos
    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    X_train_discrete = discretizer.fit_transform(X_train_scaled).astype(int)
    X_test_discrete = discretizer.transform(X_test_scaled).astype(int)
    
    # Instanciar ambiente e agente
    env_train = IrisEnvironment(X_train_discrete, y_train)
    agent = QLearningAgent(n_actions=3, alpha=0.1, gamma=0.9, epsilon=0.1)
    
    rewards_per_episode = []
    
    # --- TREINAMENTO ---
    for ep in range(n_episodes):
        state = env_train.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.choose_action(state, train=True)
            next_state, reward, done = env_train.step(action)
            agent.update_q_value(state, action, reward, next_state)
            
            state = next_state
            total_reward += reward
            
        # Calcula a recompensa média do episódio (total / amostras)
        rewards_per_episode.append(total_reward / env_train.n_samples)
        
    # --- TESTE ---
    env_test = IrisEnvironment(X_test_discrete, y_test)
    state = env_test.reset()
    done = False
    y_pred = []
    
    while not done:
        action = agent.choose_action(state, train=False) # Sem exploração no teste
        y_pred.append(action)
        next_state, _, done = env_test.step(action)
        state = next_state
        
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    return acc, cm, rewards_per_episode

# ==========================================
# 4. EXECUÇÃO DAS 30 RODADAS E RESULTADOS
# ==========================================

if __name__ == "__main__":
    print("Iniciando as 30 rodadas experimentais...")
    
    n_runs = 30
    accuracies = []
    confusion_matrices = []
    all_learning_curves = []
    
    for i in range(n_runs):
        acc, cm, learning_curve = run_experiment(n_episodes=500) # 500 episódios por rodada
        accuracies.append(acc)
        confusion_matrices.append(cm)
        all_learning_curves.append(learning_curve)
        if (i+1) % 5 == 0:
            print(f"Rodada {i+1}/{n_runs} concluída. Acurácia: {acc:.4f}")
            
    # Estatísticas
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    mean_cm = np.mean(confusion_matrices, axis=0)
    
    print("\n" + "="*40)
    print("RESULTADOS FINAIS")
    print("="*40)
    print(f"Acurácia Média: {mean_acc:.4f}")
    print(f"Desvio Padrão:  {std_acc:.4f}")
    
    # ==========================================
    # 5. GERAÇÃO DOS GRÁFICOS
    # ==========================================
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Gráfico 1: Curva de Aprendizado (Média de todas as rodadas)
    mean_learning_curve = np.mean(all_learning_curves, axis=0)
    axes[0].plot(mean_learning_curve, color='blue')
    axes[0].set_title('Curva de Aprendizado')
    axes[0].set_xlabel('Episódios')
    axes[0].set_ylabel('Recompensa Média')
    axes[0].grid(True)
    
    # Gráfico 2: Matriz de Confusão (Comparação entre classes)
    sns.heatmap(mean_cm, annot=True, fmt='.1f', cmap='Blues', ax=axes[1],
                xticklabels=['Setosa', 'Versicolor', 'Virginica'],
                yticklabels=['Setosa', 'Versicolor', 'Virginica'])
    axes[1].set_title('Matriz de Confusão Média')
    axes[1].set_xlabel('Previsto pelo Agente')
    axes[1].set_ylabel('Classe Real')
    
    # Gráfico 3: Distribuição de Acertos
    sns.histplot(accuracies, bins=10, kde=True, color='green', ax=axes[2])
    axes[2].set_title('Distribuição de Acertos (Acurácia)')
    axes[2].set_xlabel('Acurácia')
    axes[2].set_ylabel('Frequência nas 30 rodadas')
    
    plt.tight_layout()
    plt.savefig('graficos.png', dpi=300)
    print("\nOs gráficos foram salvos com sucesso no arquivo 'graficos.png'.")