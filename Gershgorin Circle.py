import numpy as np
from numpy.random import randint
from numpy.linalg import eig
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

N = 3
M = randint(-9, 9, (N, N))
M = M + M.T
T = [[True for i in range(N) if i==j] for j in range(N)]

def Gershgorin(A):
    L = A.shape[0]
    D = []
    for i in range(L):
        row = A[i]
        diag = row[i]
        other = np.concatenate((row[:i], row[i+1:]))
        rad = sum(abs(other))
        D.append([diag, rad])

    return np.array(D)


G = Gershgorin(M)
H = eig(M)
print("Original Matrix :- \n", M, '\n')
print("Eigenvalues :-\n ", H[0], '\n')
print("Eigenvectors :-\n ", H[1], '\n')

for i in G:
    c = Circle((i[0], 0), i[1], alpha=0.2, fc=np.random.random(3))
    plt.gca().add_patch(c)

plt.subplots_adjust(0.05, 0.05, 0.95, 0.95)
plt.gca().set_aspect('equal')
plt.gca().spines[['left', 'bottom']].set_position('center')
plt.gca().spines[['right', 'top']].set_visible(False)
plt.plot(H[0], np.zeros_like(H[0]), 'ok')
plt.show()



