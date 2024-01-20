import numpy as np
from numpy.linalg import matrix_power as mpow
from numpy.linalg import inv
np.set_printoptions(precision=2, suppress=True)

def Frobenius(A):
    x = np.ones(N)
    K = np.array([mpow(M, i) @ x for i in range(N)]).T
    G = inv(K) @ M @ K
    return np.concatenate((-G[:, -1], np.ones(1)))[::-1]


def Toeplitz(A: list):
    n = len(A)
    B = np.zeros((n, n))
    for i, a in enumerate(A):
        val = np.repeat(a, n-i)
        B = B + np.diag(val, -i)

    return B[:, :-1]


def Berkowitz(A):

    N = A.shape[0]
    a11 = A[0, 0]
    if N == 1:
        return [1, -a11]

    R = A[0, 1:]
    C = A[1:, 0]
    M1 = A[1:, 1:]

    carr = [1, -a11]
    for i in range(N - 1):
        p = np.linalg.matrix_power(M1, i)
        carr.append(-R @ p @ C)

    T = Toeplitz(carr)
    return T @ Berkowitz(A[1:, 1:])


N = 5
M = np.random.randint(-9, 9, (N, N))
M = M + M.T

print(Frobenius(M), '\n')
print(Berkowitz(M), '\n')