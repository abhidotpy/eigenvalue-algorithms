import numpy as np

def Power(A, init_guess=None, eps=0.001):
    L = A.shape[0]
    eig_old = np.zeros(L)
    eig_new = np.ones(L) if init_guess is None else np.array(init_guess)
    count = 0

    while not np.allclose(np.fabs(eig_new), np.fabs(eig_old), atol=eps):
        eig_old = eig_new
        C = A @ eig_old
        eig_new = C / np.linalg.norm(C)
        count += 1

    eig_val = eig_new @ A @ eig_new
    return eig_val, eig_new, count

M = np.random.randint(-9, 9, (3, 3), dtype='int64')
M = M + M.T
eig, ev, k = Power(M)

print("Original Matrix :-\n", M, '\n')
print("Largest eigenvalue :- \n", eig, '\n')
print("Corresponding eigenvector :- \n", ev, '\n')