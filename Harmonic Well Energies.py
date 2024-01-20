import numpy as np
import matplotlib.pyplot as plt


def Jacobi(A, eps=1E-3, limit=10000):

    row, col = np.shape(A)
    B = A.copy()
    R = np.eye(col)
    I = np.eye(col)
    count = 0

    while not np.allclose(np.tril(B, -1), 0, atol=eps) and count < limit:
        index = np.argmax(np.abs(np.triu(B, 1)))
        i, j = index // col, index % col

        S = np.eye(col)
        a, b = 2 * B[i, j], (B[j, j] - B[i, i])
        hyp = np.hypot(a, b)
        c = np.sqrt(0.5 * (1 + b / hyp))
        s = a / (hyp * np.sqrt(2 * (1 + b / hyp)))

        S[i, i] = S[j, j] = c
        S[i, j] = s
        S[j, i] = -s

        R = R @ S
        B = S.T @ B @ S
        count += 1

    return np.diag(B), R

def V(x):
    return 0.5 * mass * omega**2 * x**2


hbar = 1.055e-34
mass = 9.109e-31
omega = 0.01
N = 50
XL, XU = -1, 1
L = XU - XL
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

X, DX = np.linspace(XL, XU, N+2, retstep=True)
K = (2 * mass * DX ** 2) / (hbar**2)

kk1 = np.ones(len(X[1:-1]) - 1) * (-1)
kk2 = [2 + K * V(x) for x in X[1:-1]]
M = np.diag(kk1, 1) + np.diag(kk2) + np.diag(kk1, -1)

eig, ev = Jacobi(M)

print('n\tEnergy\t\t\tEigenvaue')
for i in range(11):
    E = (i + 0.5) * hbar * omega * K
    print('{}\t{:.5f}\t\t{:.5f}'.format(i, E*100, eig[i]*100))


v1, v2, v3, v4 = ev[:, 0], ev[:, 1], ev[:, 2], ev[:, 3]
v1 = np.append(np.insert(v1, 0, 0), 0)
v2 = np.append(np.insert(v2, 0, 0), 0)
v3 = np.append(np.insert(v3, 0, 0), 0)
v4 = np.append(np.insert(v4, 0, 0), 0)

ax1.plot(X, v1, 'k')
ax1.set_xticks([-1, -0.5, 0, 0.5, 1])
ax1.set_yticks([])
ax1.set_title(r'$n = 1$')

ax2.plot(X, v2, 'k')
ax2.set_xticks([-1, -0.5, 0, 0.5, 1])
ax2.set_yticks([])
ax2.set_title(r'$n = 2$')

ax3.plot(X, v3, 'k')
ax3.set_xticks([-1, -0.5, 0, 0.5, 1])
ax3.set_yticks([])
ax3.set_title(r'$n = 3$')

ax4.plot(X, v4, 'k')
ax4.set_xticks([-1, -0.5, 0, 0.5, 1])
ax4.set_yticks([])
ax4.set_title(r'$n = 4$')

plt.subplots_adjust(0.05, 0.05, 0.95, 0.95)
plt.show()
