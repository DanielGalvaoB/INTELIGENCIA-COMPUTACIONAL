import matplotlib.pyplot as plt
import numpy as np

# Dados obtidos no treino manual
w1, w2, b = 2, 2, -2

# Pontos para o gráfico
x_coords = [1, 1, -1, -1]
y_coords = [1, -1, 1, -1]
cores = ['blue' if t == 1 else 'red' for t in [1, -1, -1, -1]]

plt.figure(figsize=(8, 6))
plt.scatter(x_coords, y_coords, c=cores, s=100, edgecolors='black', label='Dados')

# Calculando a linha de decisão: w1*x1 + w2*x2 + b = 0  => x2 = (-w1*x1 - b) / w2
x_reta = np.linspace(-2, 2, 100)
y_reta = (-w1 * x_reta - b) / w2

plt.plot(x_reta, y_reta, '--k', label='Fronteira de Decisão')
plt.fill_between(x_reta, y_reta, 2, color='blue', alpha=0.1)
plt.fill_between(x_reta, y_reta, -2, color='red', alpha=0.1)

plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.title(f"Fronteira de Decisão: {w1}x1 + {w2}x2 + ({b}) = 0")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()