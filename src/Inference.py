# 库
from typing import List
import numpy as np
# from scipy.optimize import approx_fprime
from typing import List, Callable
from scipy.optimize import dual_annealing
# import pandas as pd
# from pyswarm import pso
import random
from scipy.optimize._numdiff import approx_derivative


# Calculation of Potential Energy Function U(x)
def create_Ux(n: np.ndarray, a: np.ndarray, b: float):
    indices = np.cumsum(np.concatenate([[0], n]))
    
    def Ux(x: np.ndarray) -> float:
        segment_sums = np.array([
            np.sum(x[indices[i]:indices[i+1]]) 
            for i in range(len(n))
        ])
        neighbor_product = np.sum(x[:-1] * x[1:]) if len(x) > 1 else 0.0
        u = np.dot(a, segment_sums) + b * neighbor_product
        
        return -u
    
    return Ux
# # check
# n = np.array([2, 2])
# a = np.array([1.0, 1.0])
# b = 1.0
# x = np.array([1, 1, 1, 1])

# Ux_fun = create_Ux(n, a, b)
# print(Ux_fun(x))  # Output should be -7.0

# Calculate the matrix raised to the power of $N_k - 1$ 
def get_W(n: int, a: float, b: float) -> np.ndarray:
    if n == 1:
        return np.array([[1.0, 0.0], [0.0, 1.0]])

    exp_a = np.exp(a)
    exp_b = np.exp(b)
    cosh_a = 0.5 * (exp_a + 1.0 / exp_a)
    sinh_a = exp_a - cosh_a
    aux1 = exp_b * cosh_a
    aux2 = np.sqrt(1.0 + exp_b**4 * sinh_a**2)
    lambda1N = (aux1 - 1.0 / exp_b * aux2)**(n - 1)
    lambda2N = (aux1 + 1.0 / exp_b * aux2)**(n - 1)
    aux1 = -exp_b**2 * sinh_a
    e1 = np.array([aux1 - aux2, 1.0])
    e1 /= np.linalg.norm(e1)         
    e2 = np.array([aux1 + aux2, 1.0])
    e2 /= np.linalg.norm(e2)

    return np.outer(e1, e1) * lambda1N + np.outer(e2, e2) * lambda2N

## check
# W = get_W(4, 0.0, 0.0)
# print(W)



def get_V(a: np.ndarray, b: float) -> np.ndarray:
    exp_b = np.exp(b)
    exp_a_p = np.exp(0.5 * (a[1] + a[0]))
    exp_a_m = np.exp(0.5 * (a[1] - a[0]))

    return np.array([
        [exp_b / exp_a_p, exp_a_m / exp_b],
        [1.0 / (exp_b * exp_a_m), exp_b * exp_a_p]
    ])

# a = np.array([1.0, 1.0], dtype=np.float64)
# print(get_V(a, 1.0))


def get_u(a: float) -> np.ndarray:

    exp_aux = np.exp(a/2.0)

    return np.array([
        [1.0/exp_aux],
        [exp_aux]
    ])

# # check
# u = get_u(0.0)
# print(u)


def _site_fields(n, a):
    """Expand nonempty subregion fields to the CpG chain."""
    counts = np.asarray(n)
    fields = np.asarray(a, dtype=float)
    if (counts.ndim != 1 or fields.ndim != 1 or counts.size == 0
            or counts.size != fields.size or not np.all(np.isfinite(counts))
            or np.any(counts <= 0) or np.any(counts != np.floor(counts))):
        raise ValueError("n must contain positive integer CpG counts, one per alpha")
    if not np.all(np.isfinite(fields)):
        raise ValueError("alpha must be finite")
    return np.repeat(fields, counts.astype(int))


def _log_partition_fields(h, b):
    """Two-state transfer recursion in log space; never forms exp(logZ)."""
    if not np.isfinite(b):
        raise ValueError("beta must be finite")
    log_minus, log_plus = -h[0], h[0]
    for field in h[1:]:
        next_minus = -field + np.logaddexp(log_minus + b, log_plus - b)
        next_plus = field + np.logaddexp(log_minus - b, log_plus + b)
        log_minus, log_plus = next_minus, next_plus
    return float(np.logaddexp(log_minus, log_plus))


def comp_logZ(n: np.ndarray, a: np.ndarray, b: float) -> float:
    """Compute log(Z) directly, including when Z exceeds Float64 range."""
    return _log_partition_fields(_site_fields(n, a), float(b))


def _exp_partition(log_value, name):
    # Compatibility for external callers that actually need Z or g.
    if not np.isfinite(log_value) or log_value > np.log(np.finfo(float).max):
        raise OverflowError(f"{name} exceeds float range; use its log-space function")
    return float(np.exp(log_value))


def comp_Z(n: np.ndarray, a: np.ndarray, b: float) -> float:
    """Compatibility wrapper returning Z. Internal calculations use comp_logZ."""
    return _exp_partition(comp_logZ(n, a, b), "Z")

# # check
# n = np.array([1,1,1])
# a = np.array([1.0,2.0,3.0])
# b = 1.0
# Z = comp_Z(n,a,b)
# Ux_fun = create_Ux(n,a,b)
# from itertools import product
# # Generate all rows with three columns, where each column takes a value from {-1, +1}.
# M = np.array(list(product([-1, 1], repeat=3)), dtype=int)
# def f(x):
#     return 1.0 / np.exp(Ux_fun(x))
# total = sum(f(row) for row in M)
# Z_ln = comp_logZ(n, a, b)
# print(Z)
# print(total)
# print("exp(logZ):", np.exp(Z_ln))




def get_grad_logZ(n:np.ndarray, θhat: np.ndarray) -> np.ndarray:
    def f(θ):
        return comp_logZ(n, θ[:-1], θ[-1])


    grad_logZ = approx_derivative(f, θhat, method="3-point",rel_step=1e-6)  
    return grad_logZ


# #check
# n=np.array([10])
# theta1=np.array([0.5,0.5])
# theta2=np.array([-0.5,0.5])
# print(get_grad_logZ(n,theta1),get_grad_logZ(n,theta2))
# #check
# n=np.array([1,1,1])
# theta=np.array([1.0,-1.0,1.0,0.0])
# print(get_grad_logZ(n,theta))

# def find_theta_given_dfdb_equals_1(n: List[int], θ_init: List[float]):
#     def objective(θ):
#         grad = get_grad_logZ(n, θ)
#         return (grad[-1] - 1) ** 2 

#     res = minimize(objective, θ_init, method='BFGS')
#     return res.x, res.fun
# print(find_theta_given_dfdb_equals_1([4],[0.0,5.0]))

# # check
##n=[1]
#x=[1,1,1,-1,-1,-1]^T
#θ=[1.0,-1.0]
#E[sumx]=dlnz/da
#E[sumx_n*x_{n+1}]=dlnz/db
# x = np.array([1, 1, 1, -1, -1, -1])
# E1 = np.mean(x)
# products = x[:-1] * x[1:]  # [1*1, 1*1, 1*(-1), (-1)*(-1), (-1)*(-1)]
# E2 = np.mean(products)
# print(E1)
# print(E2)
# θhat = [1.0, -1.0]         #(α1,α2,α3,β)
# get_grad_logZ([1], θhat)     # (n1,n2,n3)


# Set maximum tolerance
ETA_MAX_ABS = 5.0     

def check_boundary(θhat: np.ndarray) -> bool:
    return np.any(np.isclose(np.abs(θhat), ETA_MAX_ABS, atol=5e-2)) or np.any(np.abs(θhat) > ETA_MAX_ABS)

# # check
# print(check_boundary([1.0, 1.0, 1.0, 1.0]))  # 输出: False
# print(check_boundary([1.0, 5.0, 1.0, 1.0]))  # 输出: True

def comp_log_g(r: np.ndarray, a: np.ndarray, b: float, ap1: float, ap2: float) -> float:
    """Log of a missing-segment partition factor with modified boundaries."""
    r = np.asarray(r)
    if r.size == 0 or np.all(r == 0):
        return 0.0
    h = _site_fields(r, a)
    # The original transfer matrix splits each endpoint field in half.
    # For a one-site segment both adjustments apply to that same site.
    h[0] += 0.5 * (ap1 - a[0])
    h[-1] += 0.5 * (ap2 - a[-1])
    return _log_partition_fields(h, float(b))


def comp_g(r: np.ndarray, a: np.ndarray, b: float, ap1: float, ap2: float) -> float:
    """Compatibility wrapper; use comp_log_g for likelihoods and sampling."""
    return _exp_partition(comp_log_g(r, a, b, ap1, ap2), "g")

# # check
# r=np.array([1])
# a=np.array([1.0])

# result = comp_g(r,a, 1.0, 3.0, 3.0)
# print(result)  # 输出: 20.13532399

#Marginalized Methylation Probability
def comp_lkhd(x: np.ndarray, n: np.ndarray, a: np.ndarray, b: float) -> float:
    """Probability of a full/partial read, formed only after log normalization."""
    x = np.asarray(x, dtype=float)
    n = np.asarray(n, dtype=int)
    theta = np.r_[np.asarray(a, dtype=float), b]
    log_p = -create_Llkhd(n, x.reshape(1, -1))(theta)
    return float(np.exp(min(0.0, log_p)))




#Maximum Likelihood Function
def create_Llkhd(n: np.ndarray, xobs: np.ndarray) -> Callable[[List[float]], float]:
    n_array = np.asarray(n)
    _site_fields(n_array, np.zeros(len(n_array)))  # validate counts before casting
    n = n_array.astype(int)
    xobs = np.asarray(xobs, dtype=float)
    if xobs.ndim != 2 or xobs.shape[1] != n.sum():
        raise ValueError("xobs must be a reads-by-CpG matrix with sum(n) columns")
    if not np.all(np.isin(xobs, [-1, 0, 1])):
        raise ValueError("xobs must use -1/1 for observed states and 0 for missing")

    subid = np.repeat(np.arange(1, len(n) + 1), n)
    N = xobs.shape[1]

    def Llkhd_fun(theta: np.ndarray) -> float:
        # Validate theta: length, finiteness, and bounds
        # if theta is None or len(theta) != (len(n) + 1):
        #     raise ValueError(
        #         f"theta must have length len(n)+1={len(n)+1}, got {0 if theta is None else len(theta)}."
        #     )
        # Get parameters
        aux = 0.0
        a = theta[:-1]
        b = float(theta[-1])

        # Get energy function and partition function
        Ux = create_Ux(n, a, b)
        logZ = comp_logZ(n, a, b)

        # Initialize variables used in for loop
        ap1 = ap2 = 0.0  
        # Contribution of each observation
        for x in xobs:
            absx = np.abs(x)
            d = absx[1:] - absx[:-1]
            zerost = np.flatnonzero(d == -1) + 1
            zeroend = np.flatnonzero(d == 1) + 1
            if x[0] == 0:
                zerost = np.insert(zerost, 0, 0)
            if x[-1] == 0:
                zeroend = np.append(zeroend, N)

            for s, e in zip(zerost, zeroend):
                ap1 = a[0] if s == 0 else 2.0 * x[s - 1] * b + a[subid[s] - 1]
                ap2 = a[-1] if e == N else 2.0 * x[e] * b + a[subid[e - 1] - 1]
                b_id = subid[s:e]
                if b_id.size == 0:
                    continue
                uniq, counts = np.unique(b_id, return_counts=True)
                n_miss = counts.astype(int)
                # print("n_miss*:",n_miss)
                a_sub = a[uniq - 1]
                # print("a_sub*:",a_sub)

                aux += comp_log_g(n_miss, a_sub, b, ap1, ap2)

            # Add log of it to Llkhd
            aux += -Ux(x)

        # Return MINUS log-likelihood
        return -aux + len(xobs) * logZ

    return Llkhd_fun
# # check
# # Comparing two probability functions
# Z = comp_Z([5],[1.0],1.0)
# Ux_fun = create_Ux([5], [1.0], 1.0)
# u1=1.0/np.exp(Ux_fun(np.array([1,1,-1,1,1])))
# result_1=u1/Z
# print(result_1)
# result_2 = comp_lkhd(np.array([1, 1, -1, 1, 1]), [5], [1.0], 1.0)
# print(result_2) 
# result_3 = comp_lkhd(np.array([1, 1, 0, 1, 1]), [5], [1.0], 1.0)
# print(result_3) 
# u2=1.0/np.exp(Ux_fun(np.array([1,1,1,1,1])))
# result_4=(u1+u2)/Z
# print(result_4)

# x=np.array([0,0,1,1,0,0,0,-1,-1,0,0])
# n=np.array([5,6])
# a=np.array([1.0,2.0])
# b=2.0
# result=comp_lkhd(x,n,a,b)
# print(result)
# #check
# xobs =  [[-1,  0, 0, 1],
#          [ 0,  0, -1, 0]]
# n = [1,3]
# n_vec = np.array(n, dtype=int)
# xobs_vec = np.array(xobs, dtype=float)
# LogLike = create_Llkhd(n_vec,xobs_vec)
# theta = np.array([1.0,2.0,3.0])
# result_1=LogLike(theta)
# print(result_1)
# #比较
# u1=comp_lkhd(np.array([-1,-1, -1, 1]), [4], [1.0], 1.0)
# u2=comp_lkhd(np.array([ 1, 1, -1, 1]), [4], [1.0], 1.0)
# result_2 = -np.log(u1)-np.log(u2)
# print(result_1)
# print(result_2)


# Initialize parameters, output (a, b=0)
def est_alpha(xobs:np.ndarray) -> np.ndarray:
    phat = np.mean(xobs == 1)
    phat = np.clip(phat, 1e-8, 1 - 1e-8)
    a = 0.5 * (np.log(phat) - np.log(1.0 - phat))
    a = np.clip(a, -ETA_MAX_ABS, ETA_MAX_ABS)
    return np.array([a, 0.0])

ETA_MAX_ABS = 5.0
class EarlyStopper:
    def __init__(self, tol=1e-6, patience=20):
        """
        tol: Minimum change threshold (convergence is assumed if the change in the objective function is less than tol)
        patience: Number of consecutive times the change is less than tol before triggering early stopping
        """
        self.tol = tol
        self.patience = patience
        self.best_f = np.inf
        self.counter = 0

    def __call__(self, x, f, context):
        if abs(self.best_f - f) < self.tol:
            self.counter += 1
        else:
            self.counter = 0
        self.best_f = f

        if self.counter >= self.patience:
            return True  
        return False
def est_theta_sa1(n: np.array, xobs: np.array) -> np.ndarray:
    if sum(n) == 1:
        return est_alpha(xobs)

    L = create_Llkhd(n, xobs)
    bounds = [(-ETA_MAX_ABS, ETA_MAX_ABS)] * (len(n) + 1)
    init=np.zeros(len(bounds),dtype=float)
    early_stop = EarlyStopper(tol=1e-8, patience=30)
    result = dual_annealing(L, bounds=bounds,maxiter=10000,x0=init,restart_temp_ratio=1e-4,no_local_search=True,initial_temp= 1.0,callback=early_stop)

    return result.x

# random.seed(42)
# np.random.seed(42)
def simulated_annealing(
        func, bounds, max_iter=1000, initial_temp=100, cooling_rate=0.95, init=None, f_tol=None
):

    dim = len(bounds)

    if init is None:
        current_pos = np.array([random.uniform(b[0], b[1]) for b in bounds])
    else:
        if len(init) != dim:
            raise ValueError(f"init length should be {dim}, but received {len(init)}")
        current_pos = np.array(init, dtype=float)
        current_pos = np.clip(current_pos, [a for (a, _) in bounds], [b for (_, b) in bounds])

    current_val = func(current_pos)
    if not np.isfinite(current_val):
        raise ValueError("Initial objective is not finite")
    best_pos, best_val = current_pos.copy(), float(current_val)
    temp = initial_temp
    
    T_EPS = 1e-12
    for i in range(max_iter):
        new_pos = current_pos + np.random.normal(0, 1, dim) * temp
        new_pos = np.clip(new_pos, [b[0] for b in bounds], [b[1] for b in bounds])
        new_val = func(new_pos)
        if np.isfinite(new_val) and ((new_val < current_val) or (random.random() < np.exp(min(0.0, (current_val - new_val) / max(temp, T_EPS))))):
            current_pos, current_val = new_pos, new_val
            if new_val < best_val:
                if (f_tol is not None) and ((best_val - new_val) < f_tol):
                    best_pos, best_val = new_pos.copy(), new_val
                    break
                best_pos, best_val = new_pos.copy(), new_val

        temp *= cooling_rate

        if temp < T_EPS:
            break

    return best_pos

from scipy.optimize import minimize, OptimizeResult

def hybrid_simulated_annealing(
        func, bounds, max_iter=1000, initial_temp=100, cooling_rate=0.95,
        init=None, f_tol=None, local_search_interval=50, local_search_method="Powell"
):
    dim = len(bounds)

    if init is None:
        current_pos = np.array([random.uniform(b[0], b[1]) for b in bounds])
    else:
        if len(init) != dim:
            raise ValueError(f"init length should be {dim}, but received {len(init)}")
        current_pos = np.array(init, dtype=float)
        current_pos = np.clip(current_pos, [a for (a, _) in bounds], [b for (_, b) in bounds])

    current_val = func(current_pos)
    if not np.isfinite(current_val):
        raise ValueError("Initial objective is not finite")
    best_pos, best_val = current_pos.copy(), float(current_val)
    temp = initial_temp

    T_EPS = 1e-12
    nfev = 1  
    nit = 0   

    for i in range(max_iter):
        nit += 1
        new_pos = current_pos + np.random.normal(0, 1, dim) * temp
        new_pos = np.clip(new_pos, [b[0] for b in bounds], [b[1] for b in bounds])
        new_val = func(new_pos); nfev += 1

        if np.isfinite(new_val) and ((new_val < current_val) or (random.random() < np.exp(min(0.0, (current_val - new_val) / max(temp, T_EPS))))):
            current_pos, current_val = new_pos, new_val

            if new_val < best_val:
                if (f_tol is not None) and ((best_val - new_val) < f_tol):
                    best_pos, best_val = new_pos.copy(), new_val
                    break
                best_pos, best_val = new_pos.copy(), new_val

        if (i + 1) % local_search_interval == 0:
            res = minimize(func, current_pos, bounds=bounds, method=local_search_method)
            nfev += res.nfev if hasattr(res, "nfev") else 0
            if res.success and np.isfinite(res.fun) and res.fun < best_val:
                best_pos, best_val = res.x.copy(), res.fun
                current_pos, current_val = res.x.copy(), res.fun

        temp *= cooling_rate
        if temp < T_EPS:
            break

    result = OptimizeResult(
        x=best_pos,
        fun=best_val,
        nit=nit,
        nfev=nfev,
        message="Optimization terminated",
        success=True
    )
    return result.x
def est_theta_sa(n:np.ndarray, xobs: np.ndarray) -> np.ndarray:
    if n.sum() == 1:
        return est_alpha(xobs)

    L = create_Llkhd(n, xobs)

    bounds = [(-ETA_MAX_ABS, ETA_MAX_ABS)] * (len(n) + 1)

    result = hybrid_simulated_annealing(
    L, bounds, max_iter=1000,
    initial_temp=10, cooling_rate=0.98,
    local_search_interval=50, f_tol=1e-6
)
    return result
    

def comp_mml(n:np.ndarray, grad_logZ) -> float:
    return float(np.abs(np.round(0.5 * (1.0 + grad_logZ[:len(n)].sum() / n.sum()), 8)))


# # check
n = np.array([1,1,1])
θ = np.array([1.0,-1.0,3.0,4.0])
grad_logZ = get_grad_logZ(n,θ)
# result_mml=comp_mml(n,grad_logZ)
# print(result_mml)

def comp_nme(n: np.ndarray, θ: np.ndarray, grad_logZ: np.ndarray) -> float:
    return float(np.abs(np.round((comp_logZ(n, θ[:-1], θ[-1]) - np.dot(θ, grad_logZ)) / (n.sum() * np.log(2.0)),8)))

# result_nme=comp_nme(n,θ,grad_logZ)
# n = np.array([10])
# θ1 = np.array([0.5,0.5])
# θ2 = np.array([-0.5,0.5])
# grad_logZ1 = get_grad_logZ(n,θ1)
# grad_logZ2 = get_grad_logZ(n,θ2)
# result_nme1=comp_nme(n,θ1,grad_logZ1)
# result_nme2=comp_nme(n,θ2,grad_logZ2)
# print(result_nme)
# print(result_nme1)
# print(result_nme2)  

def comp_cmd(n: np.ndarray, θ1: np.ndarray, θ2: np.ndarray, grad_logZ1, grad_logZ2) -> float: 
    
    θγ = 0.5 * (θ1+θ2)

    logZ1 = comp_logZ(n, θ1[:-1], θ1[-1])
    logZ2 = comp_logZ(n, θ2[:-1], θ2[-1])
    logZγ = comp_logZ(n, θγ[:-1], θγ[-1])

    cmd = logZ1 + logZ2 - (np.dot(θ1, grad_logZ1) + np.dot(θ2, grad_logZ2))

    cmd /= 2.0 * logZγ - np.dot(θγ, (grad_logZ1 + grad_logZ2))

    return float(1.0 - cmd)


# Non-normalized case

def comp_cmd_unnorm(n: np.ndarray, θ1: np.ndarray, θ2: np.ndarray, grad_logZ1: np.ndarray, grad_logZ2: np.ndarray) -> float:
    θγ = 0.5 * (θ1 + θ2)
    logZ1 = comp_logZ(n, θ1[:-1], θ1[-1])
    logZ2 = comp_logZ(n, θ2[:-1], θ2[-1])
    logZγ = comp_logZ(n, θγ[:-1], θγ[-1])
    cmd = 2.0 * logZγ - np.dot(θγ, (grad_logZ1 + grad_logZ2)) - (logZ1 + logZ2 - (np.dot(θ1, grad_logZ1) + np.dot(θ2, grad_logZ2)))
    return float(cmd)



# # check
# n=np.array([20])
# θ1=θ2=np.array([1.0,1.0])
# grad_logZ1s = get_grad_logZ(n,θ1)
# grad_logZ2s = get_grad_logZ(n,θ2)
# print(comp_cmd_unnorm(n,θ1,θ2,grad_logZ1s,grad_logZ2s))
# n=[20]
# θ1=np.array([-5.0,5.0])
# θ2=np.array ([5.0,5.0])
# grad_logZ1s = get_grad_logZ(n,θ1)
# grad_logZ2s = get_grad_logZ(n,θ2)
# print(comp_cmd_unnorm(n,θ1,θ2,grad_logZ1s,grad_logZ2s))


def comp_pdr(n: np.ndarray, θ: np.ndarray) -> float: 
    total_cpgs = int(n.sum())
    a_dot_n = float(np.dot(θ[:-1], n))
    b = float(θ[-1])
    logZ = float(comp_logZ(n, θ[:-1], b))

    # log(cosh(x)) = logaddexp(x, -x) - log(2)
    log_cosh = float(np.logaddexp(a_dot_n, -a_dot_n) - np.log(2.0))
    log_num = np.log(2.0) + (total_cpgs - 1) * b + log_cosh
    return float(-np.expm1(min(0.0, log_num - logZ)))

# #check
# n=np.array([200])
# θ=np.array([5.0,5.0])
# pdr=comp_pdr(n,θ)
# print(pdr)

def comp_chalm(n: np.ndarray, θ: np.ndarray) -> float: 
    total_cpgs = int(n.sum())
    a_dot_n = float(np.dot(θ[:-1], n))
    b = float(θ[-1])
    logZ = float(comp_logZ(n, θ[:-1], b))

    # chalm = exp(-(a·n - b(N-1))) / Z
    log_num = -(a_dot_n - b * (total_cpgs - 1))
    return float(-np.expm1(min(0.0, log_num - logZ)))

# # check
# n=np.array([1])
# θ=np.array([1.0,1.0])
# chalm=comp_chalm(n,θ)
# rchalm= 1.0-1.0/(np.exp(2.0)+1.0)
# print( chalm, rchalm)

def comp_mcr(n: np.ndarray, θ: np.ndarray, grad_logZ: np.ndarray) -> float: 
    chalm = comp_chalm(n, θ)
    mml = comp_mml(n, grad_logZ)
    return float(chalm - mml)

# #check
# n = np.array([1,1,1])
# θ = np.array([1.0,-1.0,1.0,0.0])
# grad_logZ = get_grad_logZ(n,θ)
# mml=comp_mml(n,grad_logZ)
# chalm=comp_chalm(n,θ)
# rmcr=chalm-mml
# mcr=comp_mcr(n,θ,grad_logZ)
# print(mcr,rmcr)