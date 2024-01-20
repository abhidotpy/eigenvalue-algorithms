import numpy as np

def GaussElim(mat):
    A = mat.copy()
    A = A.astype('float')
    L = A.shape[0]
    det = 1

    for i in range(L):

        if len(mat[:, i][mat[:, i] != 0]) == 0:
            det = 0.0
            break

        # m = np.argmax(np.fabs(A[:, i][i:])) + i
        # A[[m, i]] = A[[i, m]]
        # if i != m:
        #    det = det * (-1)

        det = det * A[i, i]
        for j in range(0, L):
            if j != i:
                factor = A[j, i] / A[i, i]
                A[j] = A[j] - A[i] * factor

    return det

M = np.random.randint(-9, 9, (3, 3), dtype='int64')
M = M + M.T

print("Original Matrix :-\n", M, '\n')
print("Determinant obtained using Gaussian Elimination : ", GaussElim(M))
print("Determinant obtained using numpy linalg : ", np.linalg.det(M))
