import numpy as np
import matplotlib.pyplot as plt
import random

# Parâmetros do modelo
L = 20               # Tamanho da rede (20x20)
T = 2.3              # Temperatura
J = 1                # Constante de interação
n_steps = 100000     # Número de iterações

# Inicializa a rede com spins aleatórios (+1 ou -1)
def init_lattice(L):
    return np.random.choice([1, -1], size=(L, L))

# Calcula a variação de energia ao inverter o spin na posição (i, j)
def delta_energy(lattice, i, j, J):
    L = lattice.shape[0]
    total = (lattice[(i+1)%L, j] + lattice[(i-1)%L, j] +
             lattice[i, (j+1)%L] + lattice[i, (j-1)%L])
    dE = 2 * J * lattice[i, j] * total
    return dE

# Algoritmo de Metropolis para atualizar a rede
def metropolis(lattice, T, n_steps, J):
    L = lattice.shape[0]
    for _ in range(n_steps):
        i = random.randint(0, L-1)
        j = random.randint(0, L-1)
        dE = delta_energy(lattice, i, j, J)
        if dE < 0 or random.random() < np.exp(-dE/T):
            lattice[i, j] *= -1
    return lattice

# Calcula a magnetização média
def magnetization(lattice):
    return np.abs(np.sum(lattice)) / (lattice.shape[0]**2)

if __name__ == "__main__":
    lattice = init_lattice(L)
    lattice = metropolis(lattice, T, n_steps, J)
    mag = magnetization(lattice)
    print(f"Magnetização final: {mag:.4f}")

    # Visualiza a rede de spins
    plt.imshow(lattice, cmap='bwr')
    plt.title(f"Modelo de Ising 2D (T={T})")
    plt.colorbar(label='Spin')
    plt.show()
