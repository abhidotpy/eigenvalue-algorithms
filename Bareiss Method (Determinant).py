import numpy as np


def Bareiss(mat):
    L = mat.shape[0]
    det = 1
    A = mat.copy()

    for k in range(L):

        if A[k, k] == 0:
            det = 1
            m = np.argmax(np.fabs(A[:, k][k:])) + k
            A[[m, k]] = A[[k, m]]
            if k != m:
                det = det * (-1)

        for i in range(k + 1, L):
            for j in range(k + 1, L):
                A[i, j] = (A[i, j] * A[k, k] - A[i, k] * A[k, j])
                if k > 0:
                    A[i, j] = A[i, j] / A[k - 1, k - 1]

    return det * A[-1, -1]


M = np.random.randint(-9, 9, (3, 3), dtype='int64')
M = M + M.T

print("Original Matrix :-\n", M)
print("Determinant obtained using Bareiss Method : ", Bareiss(M))
print("Determinant obtained using numpy linalg : ", np.linalg.det(M))
