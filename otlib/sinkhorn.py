import numpy as np


def sinkhorn_marg_violation(a:np.array,b:np.array,C:np.array, eps:float, prec:float):
    """
    Sinkhorn's algorithm with marginal violation
    a,b: mass distribution
    C: cost matrix
    eps: precision used for the stopping criterion.
    """
    i_n = np.ones_like(a)
    i_m = np.ones_like(b)

    # Initialisation
    # Gibbs kernel
    K = np.exp(-C/eps)
    # Iterates
    u = a/(K@i_m)
    v = b/(K.T@u)
    # Plan
    P = np.diag(u)@K@np.diag(v)
    # Marginal violation
    marg_a = np.linalg.norm(P@i_n-a, 1)
    marg_b = np.linalg.norm(P.T@i_m-b,1)
    marg_v = max(marg_a,marg_b)
    # Historical
    hist_marg = [[marg_a],[marg_b],[marg_v]]
    
    # Iteration
    while marg_v > prec:
        # Update
        u = a/(K@v)
        v = b/(K.T@u)
        P = np.diag(u)@K@np.diag(v)
        # Marginal violation
        marg_a = np.linalg.norm(P@i_n-a, 1)
        marg_b = np.linalg.norm(P.T@i_m-b,1)
        marg_v = max(marg_a,marg_b)
        hist_marg[0].append(marg_a)
        hist_marg[1].append(marg_b)
        hist_marg[2].append(marg_v)
    return P, hist_marg



def soft_min_col(A, eps):
    return -eps*np.log(np.sum(np.exp(-A/eps), axis=0))

def soft_min_row(A, eps):
    return -eps*np.log(np.sum(np.exp(-A/eps), axis=1))

def matrix_S(f,g,C):
    n,m = C.shape
    S = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            S[i,j] = C[i,j] - f[i] - g[j]
    return S


def sinkhorn_log_dom(a:np.array,b:np.array,C:np.array, eps:float, prec:float):
    """
    Sinkhorn's algorithm with log domain version
    a,b: mass distribution
    C: cost matrix
    eps: precision used for the stopping criterion.
    """
    i_n = np.ones_like(a)
    i_m = np.ones_like(b)
    log_a = np.log(a)
    log_b = np.log(b)

    # Initialisation
    # Gibbs kernel
    K = np.exp(-C/eps)
    # Iterates
    f = eps*log_a+soft_min_col(C-np.ones_like(C),eps)
    g = eps*log_b+soft_min_row(C-f*i_m.T,eps)
    # Plan
    P = np.exp(f/eps)*K*np.exp(g/eps)
    # Marginal violation
    marg_a = np.linalg.norm(P@i_n-a, 1)
    marg_b = np.linalg.norm(P.T@i_m-b,1)
    marg_v = max(marg_a,marg_b)
    # Historical
    hist_marg = [[marg_a],[marg_b],[marg_v]]
    
    # Iteration
    while marg_v > prec:
        # Update
        S = matrix_S(f,g,C)
        f += soft_min_row(S,eps) + eps*log_a
        S = matrix_S(f,g,C)
        g += soft_min_col(S,eps) + eps*log_b
        P = np.exp(f/eps)*K*np.exp(g/eps)
        # Marginal violation
        marg_a = np.linalg.norm(P@i_n-a, 1)
        marg_b = np.linalg.norm(P.T@i_m-b,1)
        marg_v = max(marg_a,marg_b)
        hist_marg[0].append(marg_a)
        hist_marg[1].append(marg_b)
        hist_marg[2].append(marg_v)
    
    return P, marg_v


