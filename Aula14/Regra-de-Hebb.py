import numpy as np

# 1. Definição do Dataset (baseado na sua imagem)
# Entradas: [x1, x2]
entradas = np.array([
    [1, 1],
    [1, -1],
    [-1, 1],
    [-1, -1]
])

# Alvos (Targets)
targets = np.array([1, -1, -1, -1])

# 2. Passo 1: Inicializar pesos e bias com zero
w1 = 0
w2 = 0
b = 0

print("Iniciando Treinamento...")
print("-" * 30)

# 3. Passo 2: Iterar no conjunto de treinamento (1 Época)
for i in range(len(entradas)):
    x1 = entradas[i][0]
    x2 = entradas[i][1]
    t = targets[i]
    
    # Passo 2.1 e 2.2: Atualizar pesos e bias
    w1 = w1 + (x1 * t)
    w2 = w2 + (x2 * t)
    b = b + t
    
    print(f"Amostra {i+1}: x=({x1},{x2}) t={t}")
    print(f"Pesos atuais: w1={w1}, w2={w2}, b={b}")
    print("-" * 30)

print("\nRESULTADO FINAL:")
print(f"w1 Final: {w1}")
print(f"w2 Final: {w2}")
print(f"b Final: {b}")

# 4. Teste de Validação
print("\nTESTE DE VALIDAÇÃO (y = sign(w1x1 + w2x2 + b)):")
for i in range(len(entradas)):
    soma = (w1 * entradas[i][0]) + (w2 * entradas[i][1]) + b
    y = 1 if soma >= 0 else -1
    status = "OK" if y == targets[i] else "ERRO"
    print(f"Entrada: {entradas[i]} -> Saída: {y} (Alvo: {targets[i]}) [{status}]")