from typing import List
import numpy as np
from scipy.optimize import approx_fprime
from typing import List, Callable
from scipy.optimize import minimize
from Inference import *
import pandas as pd
from statsmodels.stats.multitest import multipletests
import os
from itertools import combinations
from math import comb  


OBSERVED_METRIC_NAMES = (
    'MML', 'NME', 'PDR', 'CHALM', 'MCR', 'FDRP', 'qFDRP',
    'RC', 'CC', 'NRC', 'NCC',
)


def _unmat_permutation_plan(m1, m2, Lmax):
    """Keep the original combination order and NumPy random-number stream."""
    L = comb(m1 + m2, m1)
    exact = L < Lmax
    comb_iter = combinations(range(m1 + m2), m1)
    if exact:
        return list(comb_iter), exact
    ind_subset = np.random.choice(L, Lmax, replace=False)
    return [ids for idx, ids in enumerate(comb_iter) if idx in ind_subset], exact


def _mat_permutation_plan(m, Lmax):
    """Keep the original sign ranks, including repeated Monte Carlo draws."""
    exact = 2**m < Lmax
    if exact:
        return list(range(2**m)), exact
    return [random.randint(0, 2**m - 1) for _ in range(Lmax)], exact


def _permutation_pvalue(stat_val, perms, exact, two_sided=True, tie_tolerance=0.0):
    values = np.asarray(perms)
    threshold = stat_val
    if two_sided:
        values = np.abs(values)
        threshold = abs(stat_val)
    extreme = values >= threshold
    if tie_tolerance:
        extreme |= np.isclose(values, threshold, rtol=tie_tolerance, atol=tie_tolerance)
    count = np.count_nonzero(extreme)
    if exact:
        return float(count / len(values))
    return float((1.0 + count) / (1.0 + len(values)))


def _validate_observed_values(values1, values2, Lmax, test_types, paired=False):
    values1 = np.asarray(values1, dtype=float)
    values2 = np.asarray(values2, dtype=float)
    if values1.ndim != 2 or values2.ndim != 2:
        raise ValueError('Observed values must be sample-by-metric matrices.')
    if values1.shape[0] == 0 or values2.shape[0] == 0:
        raise ValueError('Each group must contain at least one sample.')
    if values1.shape[1] == 0 or values1.shape[1] != values2.shape[1]:
        raise ValueError('Groups must have the same nonzero number of metric columns.')
    if paired and values1.shape != values2.shape:
        raise ValueError('Paired groups must contain the same number of samples.')
    if isinstance(Lmax, (bool, np.bool_)) or not isinstance(Lmax, (int, np.integer)) or Lmax < 1:
        raise ValueError('Lmax must be a positive integer.')
    names = list(OBSERVED_METRIC_NAMES) if test_types is None else list(test_types)
    if len(names) != values1.shape[1]:
        raise ValueError('test_types must give one name for each metric column.')
    allowed = {name.lower() for name in OBSERVED_METRIC_NAMES}
    if any(not isinstance(name, str) or name.lower() not in allowed for name in names):
        raise ValueError('Unknown observed metric name.')
    if len({name.lower() for name in names}) != len(names):
        raise ValueError('Metric column names must be unique (ignoring case).')
    return values1, values2, names


def unmat_observed_tests(values1, values2, Lmax=1000, test_types=None):
    """Compare observed sample metrics using the existing unpaired label test.

    Rows are biological samples; columns are metrics named by ``test_types``.
    Names are case-insensitive and preserved in the result. If omitted, all
    eleven ``OBSERVED_METRIC_NAMES`` columns are required in that order.
    Any nonfinite sample invalidates its entire metric column, without dropping
    samples. Two-sided ties use a 1e-12 absolute/relative floating-point tolerance.
    """
    values1, values2, names = _validate_observed_values(
        values1, values2, Lmax, test_types)
    plans, exact = _unmat_permutation_plan(len(values1), len(values2), Lmax)
    pooled = np.concatenate((values1, values2), axis=0)
    results = {}
    for column, name in enumerate(names):
        if not np.all(np.isfinite(pooled[:, column])):
            results[name] = (np.nan, np.nan)
            continue
        stat_val = float(np.mean(values1[:, column]) - np.mean(values2[:, column]))
        perms = []
        for ids in plans:
            ids = list(ids)
            first = pooled[ids, column]
            second = np.delete(pooled[:, column], ids, axis=0)
            perms.append(float(np.mean(first) - np.mean(second)))
        results[name] = (stat_val, _permutation_pvalue(
            stat_val, perms, exact, tie_tolerance=1e-12))
    return results


def mat_observed_tests(values1, values2, Lmax=1000, test_types=None):
    """Compare observed paired sample metrics using the existing sign test.

    Rows must be matched in the same order. Column names, nonfinite handling,
    and floating-point tie handling follow ``unmat_observed_tests``. The shared
    sign plan uses every sign assignment when 2**n_pairs < Lmax; otherwise it
    draws Lmax sign assignments with replacement, as in ``mat_tests``.
    """
    values1, values2, names = _validate_observed_values(
        values1, values2, Lmax, test_types, paired=True)
    js, exact = _mat_permutation_plan(len(values1), Lmax)
    results = {}
    for column, name in enumerate(names):
        if not (np.all(np.isfinite(values1[:, column]))
                and np.all(np.isfinite(values2[:, column]))):
            results[name] = (np.nan, np.nan)
            continue
        diffs = values1[:, column] - values2[:, column]
        stat_val = float(sum(diffs) / len(diffs))
        perms = comp_mat_perm_stats(diffs, js)
        results[name] = (stat_val, _permutation_pvalue(
            stat_val, perms, exact, tie_tolerance=1e-12))
    return results


def get_all_grad_logZs(n: np.ndarray, θs:np.ndarray) -> np.ndarray:
    if θs.ndim == 1:
        θs = θs.reshape(1, -1)
    return np.array([get_grad_logZ(n, θ) for θ in θs], dtype=float)


# # check
# n=np.array([5])
# θs=np.array([[1.0,2.0],[3.0,4.0],[-1.0,-2.0]])
# print(get_all_grad_logZs(n, θs))

#Compute the difference in mean test statistics between two unpaired samples
def comp_unmat_stat_mml(n: np.ndarray, grad_logZ1s: np.ndarray, grad_logZ2s:np.ndarray) ->float:
    mml1 = np.mean([comp_mml(n, g) for g in grad_logZ1s])
    mml2 = np.mean([comp_mml(n, g) for g in grad_logZ2s])
    return float(mml1 - mml2)

# # check
# n=np.array([5])
# θ1s=np.array([[1.0,1.0],[1.0,1.0]])
# θ2s=np.array([[1.0,1.0],[1.0,1.0]])
# grad_logZ1s =get_all_grad_logZs(n, θ1s)
# grad_logZ2s = get_all_grad_logZs(n, θ2s)
# print(comp_unmat_stat_mml(n, grad_logZ1s, grad_logZ2s))


def comp_unmat_perm_stat_mml(n:np.ndarray, grad_logZ1s: np.ndarray, grad_logZ2s: np.ndarray, perm_ids: List[int]) -> float:
    grad_logZ2sp = np.concatenate((grad_logZ1s, grad_logZ2s),axis=0)
    grad_logZ1sp = grad_logZ2sp[perm_ids]
    grad_logZ2sp = np.delete(grad_logZ2sp, perm_ids, axis = 0)

    return comp_unmat_stat_mml(n, grad_logZ1sp, grad_logZ2sp)



# # check
# n=np.array([10])
# θ1s = np.array([[0.5, 0.5] for _ in range(5)])
# θ2s = np.array([[-0.5, 0.5] for _ in range(5)])

# perm_ids = [0,2,4,6,8]
# grad_logZ1s = get_all_grad_logZs(n, θ1s)
# grad_logZ2s = get_all_grad_logZs(n, θ2s)

# print(comp_unmat_perm_stat_mml(n, grad_logZ1s, grad_logZ2s, perm_ids))


def comp_unmat_stat_nme(n,  θ1s, θ2s, grad_logZ1s, grad_logZ2s) -> float:
    
    if θ1s.ndim == 1: θ1s = θ1s.reshape(1, -1)
    if θ2s.ndim == 1: θ2s = θ2s.reshape(1, -1)
    if grad_logZ1s.ndim == 1: grad_logZ1s = grad_logZ1s.reshape(1, -1)
    if grad_logZ2s.ndim == 1: grad_logZ2s = grad_logZ2s.reshape(1, -1)

    nme1 = np.mean([comp_nme(n, θ, g) for θ, g in zip(θ1s, grad_logZ1s)])
    nme2 = np.mean([comp_nme(n, θ, g) for θ, g in zip(θ2s, grad_logZ2s)])
    return float(nme1 - nme2)

# # check
# n=np.array([5])
# θ1s = np.array([[0.0,0.0],[0.0,0.0]])
# θ2s = np.array([[1.0,1.0],[1.0,1.0]])
# grad_logZ1s = get_all_grad_logZs(n, θ1s)
# grad_logZ2s = get_all_grad_logZs(n, θ2s)
# print(comp_unmat_stat_nme(n,θ1s,θ2s,grad_logZ1s,grad_logZ2s))


def comp_unmat_perm_stat_nme(n, θ1s, θ2s, grad_logZ1s, grad_logZ2s, perm_ids: List[int]) -> float:
    θ2sp = np.concatenate((θ1s,θ2s))
    θ1sp = θ2sp[perm_ids]
    θ2sp = np.delete(θ2sp, perm_ids, axis=0)

    grad_logZ2sp = np.concatenate((grad_logZ1s, grad_logZ2s))
    grad_logZ1sp = grad_logZ2sp[perm_ids]
    grad_logZ2sp = np.delete(grad_logZ2sp, perm_ids, axis=0)

    return comp_unmat_stat_nme(n, θ1sp, θ2sp, grad_logZ1sp, grad_logZ2sp)
# # check
# n = np.array([10])
# θ1s = np.array([[0.5, 0.5] for _ in range(5)])
# θ2s = np.array([[-0.5, 0.5] for _ in range(5)])
# perm_ids = [0,2,4,6,8]
# grad_logZ1s = get_all_grad_logZs(n, θ1s)
# grad_logZ2s = get_all_grad_logZs(n, θ2s)
# print(comp_unmat_perm_stat_nme(n, θ1s, θ2s, grad_logZ1s, grad_logZ2s, perm_ids))


def comp_unmat_stat_pdm(n:np.ndarray,  θ1s:np.ndarray, θ2s:np.ndarray, grad_logZ1s:np.ndarray, grad_logZ2s:np.ndarray) -> float:
    cmds = []
    
    for s1 in range(len(θ1s)):
        for s2 in range(len(θ2s)):
            cmds.append(comp_cmd(n, θ1s[s1], θ2s[s2], grad_logZ1s[s1], grad_logZ2s[s2]))

    return float(np.mean(cmds))


# # check
# n=np.array([5])
# θ1s = np.array([[-1.0,1.0],[-1.0,1.0]])
# θ2s = np.array([[1.0,1.0],[1.0,1.0]])
# grad_logZ1s = get_all_grad_logZs(n, θ1s)
# grad_logZ2s = get_all_grad_logZs(n, θ2s)
# print(comp_unmat_stat_pdm(n,θ1s,θ2s,grad_logZ1s,grad_logZ2s))

def comp_unmat_perm_stat_pdm(n:np.ndarray,  θ1s:np.ndarray, θ2s:np.ndarray, grad_logZ1s:np.ndarray, grad_logZ2s:np.ndarray, perm_ids: List[int]) -> float:
    θ2sp = np.concatenate((θ1s,θ2s))
    θ1sp = θ2sp[perm_ids]
    θ2sp = np.delete(θ2sp, perm_ids, axis=0)

    grad_logZ2sp = np.concatenate((grad_logZ1s, grad_logZ2s))
    grad_logZ1sp = grad_logZ2sp[perm_ids]
    grad_logZ2sp = np.delete(grad_logZ2sp, perm_ids, axis=0)

    return comp_unmat_stat_pdm(n, θ1sp, θ2sp, grad_logZ1sp, grad_logZ2sp)

# # check
# n = np.array([10])
# θ1s = np.array([[0.5, 0.5] for _ in range(5)])
# θ2s = np.array([[-0.5, 0.5] for _ in range(5)])
# perm_ids = [0,2,4,6,8]
# grad_logZ1s = get_all_grad_logZs(n, θ1s)
# grad_logZ2s = get_all_grad_logZs(n, θ2s)
# print(comp_unmat_perm_stat_pdm(n, θ1s, θ2s, grad_logZ1s, grad_logZ2s, perm_ids))


#T_PDR
def comp_unmat_stat_pdr(n: np.ndarray, θ1s: np.ndarray, θ2s: np.ndarray) -> float:
    pdr1 =  np.mean([comp_pdr(n, θ) for θ in θ1s]) 
    pdr2 =  np.mean([comp_pdr(n, θ) for θ in θ2s]) 
    return float(pdr1 - pdr2)

# # check
# n=np.array([5])
# θ1s =np.array( [[0.0,0.0],[1.0,1.0]])
# θ2s =np.array( [[1.0,1.0],[1.0,1.0]])
# r1=comp_unmat_stat_pdr(n,θ1s,θ2s)
# r2=0.5*(comp_pdr(n,[0.0,0.0])+comp_pdr(n,[1.0,1.0]))-comp_pdr(n,[1.0,1.0])
# print(r1,r2)

#T_PDR
def comp_unmat_perm_stat_pdr(n: np.ndarray, θ1s: np.ndarray, θ2s: np.ndarray, perm_ids: List[int]) -> float:
    θ2sp = np.concatenate((θ1s, θ2s), axis=0)
    θ1sp = θ2sp[perm_ids]
    θ2sp = np.delete(θ2sp, perm_ids, axis=0)
    return comp_unmat_stat_pdr(n, θ1sp, θ2sp)

# # check
# n=np.array([5])
# θ1s =np.array( [[0.0,0.0],[0.0,0.0]])
# θ2s =np.array( [[1.0,1.0],[1.0,1.0]])
# perm_ids = [0,2]
# print(comp_unmat_perm_stat_pdr(n, θ1s,θ2s, perm_ids))

#T_CHALM
def comp_unmat_stat_chalm(n: np.ndarray, θ1s: np.ndarray, θ2s: np.ndarray) -> float:
    chalm1 =  np.mean([comp_chalm(n, θ) for θ in θ1s])
    chalm2 =  np.mean([comp_chalm(n, θ) for θ in θ2s]) 
    return float(chalm1 - chalm2)

# # check
# n=np.array([5])
# θ1s =np.array( [[0.0,0.0],[0.0,0.0]])
# θ2s =np.array( [[1.0,1.0],[1.0,1.0]])
# s1=comp_unmat_stat_chalm(n,θ1s,θ2s)
# s2=comp_chalm(n,[0.0,0.0])-comp_chalm(n,[1.0,1.0])
# print(s1,s2)


#T_CHALM
def comp_unmat_perm_stat_chalm(n: np.ndarray, θ1s: np.ndarray, θ2s: np.ndarray, perm_ids:List[int]) -> float:
    θ2sp = np.concatenate((θ1s, θ2s), axis=0)
    θ1sp = θ2sp[perm_ids]
    θ2sp = np.delete(θ2sp, perm_ids, axis=0)
    return comp_unmat_stat_chalm(n, θ1sp, θ2sp)

# #check
# n=np.array([5])
# θ1s =np.array( [[0.0,0.0],[0.0,0.0]])
# θ2s =np.array( [[1.0,1.0],[1.0,1.0]])
# perm_ids = [0,2]
# print(comp_unmat_perm_stat_chalm(n, θ1s,θ2s, perm_ids))

#T_MCR
def comp_unmat_stat_mcr(n: np.ndarray, θ1s: np.ndarray, θ2s: np.ndarray,
                        grad_logZ1s: np.ndarray, grad_logZ2s: np.ndarray) -> float:
    tchalm = comp_unmat_stat_chalm(n, θ1s, θ2s)
    tmml = comp_unmat_stat_mml(n, grad_logZ1s, grad_logZ2s)
    return tchalm - tmml
    
# # check
# n=np.array([5])
# θ1s =np.array( [[0.0,0.0],[0.0,0.0]])
# θ2s =np.array( [[1.0,1.0],[1.0,1.0]])
# grad_logZ1s = get_all_grad_logZs(n,θ1s)
# grad_logZ2s= get_all_grad_logZs(n,θ2s)
# m1 = comp_unmat_stat_mcr(n,θ1s,θ2s,grad_logZ1s, grad_logZ2s)
# m2=comp_mcr(n,[0.0,0.0],get_grad_logZ(n, [0.0,0.0]))-comp_mcr(n,[1.0,1.0],get_grad_logZ(n, [1.0,1.0]))
# print(m1,m2)

#T_MCR
def comp_unmat_perm_stat_mcr(n: np.ndarray, θ1s: np.ndarray, θ2s: np.ndarray,
                             grad_logZ1s: np.ndarray, grad_logZ2s: np.ndarray,
                             perm_ids: List[int]) -> float:
    ptchalm = comp_unmat_perm_stat_chalm(n, θ1s, θ2s, perm_ids)
    ptmml = comp_unmat_perm_stat_mml(n, grad_logZ1s, grad_logZ2s, perm_ids)
    return ptchalm - ptmml
    
# # check
# n=np.array([5,5])
# θ1s =np.array( [[0.0,1.0,0.0],[0.0,1.0,0.0]])
# θ2s =np.array( [[1.0,1.0,1.0],[1.0,1.0,1.0]])
# perm_ids = [0,2]
# grad_logZ1s = get_all_grad_logZs(n,θ1s)
# grad_logZ2s= get_all_grad_logZs(n,θ2s)
# print(comp_unmat_perm_stat_mcr(n,θ1s,θ2s,grad_logZ1s, grad_logZ2s, perm_ids))



##Unpaired group comparison (M1 != M2).
def unmat_tests(n, θ1s, θ2s, Lmax=1000,test_types=None):
    """
    Unpaired group comparison (M1 != M2).
    test_types: list of strings, default is None (computes all).
                Options: ['mml', 'nme', 'pdm', 'pdr', 'chalm', 'mcr']
    """
    valid_types = {'mml', 'nme', 'pdm', 'pdr', 'chalm', 'mcr'}
    if test_types is None:
        test_types = list(valid_types)
    else:
        test_types = [t for t in test_types if t in valid_types]   

    need_grads = any(t in test_types for t in ['mml', 'nme', 'pdm', 'mcr'])
    
    if need_grads:
        grad_logZ1s = get_all_grad_logZs(n, θ1s)
        grad_logZ2s = get_all_grad_logZs(n, θ2s)
    else:
        grad_logZ1s = grad_logZ2s = None

    diff_stats = {}
    if 'mml' in test_types:
        diff_stats['mml'] = comp_unmat_stat_mml(n, grad_logZ1s, grad_logZ2s)
    if 'nme' in test_types:
        diff_stats['nme'] = comp_unmat_stat_nme(n, θ1s, θ2s, grad_logZ1s, grad_logZ2s)
    if 'pdm' in test_types:
        diff_stats['pdm'] = comp_unmat_stat_pdm(n, θ1s, θ2s, grad_logZ1s, grad_logZ2s)
    if 'pdr' in test_types:
        diff_stats['pdr'] = comp_unmat_stat_pdr(n, θ1s, θ2s)
    if 'chalm' in test_types:
        diff_stats['chalm'] = comp_unmat_stat_chalm(n, θ1s, θ2s)
    if 'mcr' in test_types:
        diff_stats['mcr'] = comp_unmat_stat_mcr(n, θ1s, θ2s, grad_logZ1s, grad_logZ2s)

    
    comb_iter_used, exact = _unmat_permutation_plan(len(θ1s), len(θ2s), Lmax)

    perm_results = {t: [] for t in test_types}

    for x in comb_iter_used:
        perm_ids = list(x)
        if 'mml' in test_types:
            perm_results['mml'].append(comp_unmat_perm_stat_mml(n, grad_logZ1s, grad_logZ2s, perm_ids))
        if 'nme' in test_types:
            perm_results['nme'].append(comp_unmat_perm_stat_nme(n, θ1s, θ2s, grad_logZ1s, grad_logZ2s, perm_ids))
        if 'pdm' in test_types:
            perm_results['pdm'].append(comp_unmat_perm_stat_pdm(n, θ1s, θ2s, grad_logZ1s, grad_logZ2s, perm_ids))
        if 'pdr' in test_types:
            perm_results['pdr'].append(comp_unmat_perm_stat_pdr(n, θ1s, θ2s, perm_ids))
        if 'chalm' in test_types:
            perm_results['chalm'].append(comp_unmat_perm_stat_chalm(n, θ1s, θ2s, perm_ids))
        if 'mcr' in test_types:
            perm_results['mcr'].append(comp_unmat_perm_stat_mcr(n, θ1s, θ2s, grad_logZ1s, grad_logZ2s, perm_ids))
    
    results = {}
    
    for t in test_types:
        stat_val = diff_stats[t]
        perms = np.array(perm_results[t])
        pval = _permutation_pvalue(stat_val, perms, exact, two_sided=t != 'pdm')
        
        results[t] = (stat_val, float(pval))

    return results
    
   
# #check
# n = np.array([10])
# θ1s = np.array([[0.5, 0.5] for _ in range(5)])
# θ2s = np.array([[-0.5, 0.5] for _ in range(5)])
# print(unmat_tests(n, θ1s, θ2s,test_types=['pdr', 'mcr']))



##Difference statistics for matched pair i
def comp_mat_diff_mml(n, grad_logZ1s, grad_logZ2s)->np.ndarray:

    diffs = []
    
    # Iterate over the pairs of gradients and compute the MML difference
    for grad1, grad2 in zip(grad_logZ1s, grad_logZ2s):
        mml_diff = comp_mml(n, grad1) - comp_mml(n, grad2)
        diffs.append(mml_diff)
    return diffs

  
def comp_mat_diff_nme(n, θ1s, θ2s, grad_logZ1s, grad_logZ2s):
    diffs = []
    for θ1, θ2, grad1, grad2 in zip(θ1s, θ2s, grad_logZ1s, grad_logZ2s):
        nme_diff = comp_nme(n, θ1, grad1) - comp_nme(n, θ2, grad2)
        diffs.append(nme_diff)
    return diffs



def comp_mat_diff_pdr(n, θ1s, θ2s):
    diffs = []
    for θ1, θ2 in zip(θ1s, θ2s):
        pdr_diff = comp_pdr(n, θ1) - comp_pdr(n, θ2)
        diffs.append(pdr_diff)
    return diffs

def comp_mat_diff_chalm(n, θ1s, θ2s):
    diffs = []
    for θ1, θ2 in zip(θ1s, θ2s):
        chalm_diff = comp_chalm(n, θ1) - comp_chalm(n, θ2)
        diffs.append(chalm_diff)
    return diffs

    
def comp_mat_diff_mcr(n, θ1s, θ2s, grad_logZ1s, grad_logZ2s):
    diffs = []
    for θ1, θ2, grad1, grad2 in zip(θ1s, θ2s, grad_logZ1s, grad_logZ2s):
        mcr_diff = comp_mcr(n, θ1, grad1) - comp_mcr(n, θ2, grad2)
        diffs.append(mcr_diff)
    return diffs

#or
# def comp_mat_diff_mcr(n, θ1s, θ2s, grad_logZ1s, grad_logZ2s):   
#     dchalm = comp_mat_diff_chalm(n, θ1s, θ2s)
#     dmml = comp_mat_diff_mml(n, grad_logZ1s, grad_logZ2s)
#     return dchalm-dmml

# # check
# n=np.array([5])
# θ1s=np.array([[0.0,0.0],[0.0,0.0]])
# θ2s=np.array([[1.0,1.0],[1.0,1.0]])
# grad_logZ1s =get_all_grad_logZs(n, θ1s)
# grad_logZ2s = get_all_grad_logZs(n, θ2s)
# print("mml:",comp_mat_diff_mml(n,grad_logZ1s,grad_logZ2s),"nme:",comp_mat_diff_nme(n,θ1s,θ2s,grad_logZ1s,grad_logZ2s),"pdr:",comp_mat_diff_pdr(n,θ1s,θ2s),"chalm:",comp_mat_diff_chalm(n,θ1s,θ2s),"mcr:",comp_mat_diff_mcr(n,θ1s,θ2s,grad_logZ1s,grad_logZ2s))


def comp_mat_stat_pdm(n, θ1s, θ2s, grad_logZ1s, grad_logZ2s):
    cmds = 0.0
    for θ1, θ2, grad1, grad2 in zip(θ1s, θ2s, grad_logZ1s, grad_logZ2s):
        cmds += comp_cmd(n, θ1, θ2, grad1, grad2)
    return cmds / len(θ1s)

# # check
# n=np.array([5])
# θ1s=np.array([[-1.0,1.0],[-1.0,1.0]])
# θ2s=np.array([[1.0,1.0],[1.0,1.0]])
# grad_logZ1s = [get_grad_logZ(n,θ1) for θ1 in θ1s]
# grad_logZ2s = [get_grad_logZ(n,θ2) for θ2 in θ2s]
# print(comp_mat_stat_pdm(n, θ1s, θ2s, grad_logZ1s, grad_logZ2s))



def comp_mat_j_stat(diffs, j):
    diffs_perm = 0.0
    for i in range(len(diffs)):
        diffs_perm += diffs[i] if (j & 1) else -diffs[i]
        j >>= 1
    return diffs_perm/len(diffs)

# # check
# n = np.array([10])
# θ1s = np.array([[0.5, 0.5] for _ in range(5)])
# θ2s = np.array([[-0.5, 0.5] for _ in range(5)])
# grad_logZ1s = [get_grad_logZ(n,θ1) for θ1 in θ1s]
# grad_logZ2s = [get_grad_logZ(n,θ2) for θ2 in θ2s]
# mml_diffs = comp_mat_diff_mml(n, grad_logZ1s, grad_logZ2s)
# print(comp_mat_j_stat(mml_diffs, 0))





def comp_mat_perm_stats(diffs, js):
    return [comp_mat_j_stat(diffs, j) for j in js]

# # check
# n = np.array([10])
# θ1s = np.array([[0.5, 0.5] for _ in range(2)])
# θ2s = np.array([[-0.5, 0.5] for _ in range(2)])
# grad_logZ1s = [get_grad_logZ(n,θ1) for θ1 in θ1s]
# grad_logZ2s = [get_grad_logZ(n,θ2) for θ2 in θ2s]
# mml_diffs = comp_mat_diff_mml(n, grad_logZ1s, grad_logZ2s)
# js = list(range(2**len(θ1s)))
# print(comp_mat_perm_stats(mml_diffs,js))






def mat_tests(n, θ1s, θ2s, Lmax=1000, test_types=None):
    """
    配对样本比较。
    test_types: list of strings, default is None (computes all).
                Options: ['mml', 'nme', 'pdm', 'pdr', 'chalm', 'mcr']
    """
    valid_types = {'mml', 'nme', 'pdm', 'pdr', 'chalm', 'mcr'}
    if test_types is None:
        test_types = list(valid_types)
    else:
        test_types = [t for t in test_types if t in valid_types]

    # 1. 优化：仅在需要时计算梯度
    need_grads = any(t in test_types for t in ['mml', 'nme', 'pdm', 'mcr'])

    if need_grads:
        grad_logZ1s = get_all_grad_logZs(n, θ1s)
        grad_logZ2s = get_all_grad_logZs(n, θ2s)
    else:
        grad_logZ1s = grad_logZ2s = None

    # 2. 计算差异统计量 (Difference Statistics)
    diff_map = {} 
    results = {}

    if 'mml' in test_types:
        diff_map['mml'] = comp_mat_diff_mml(n, grad_logZ1s, grad_logZ2s)
    if 'nme' in test_types:
        diff_map['nme'] = comp_mat_diff_nme(n, θ1s, θ2s, grad_logZ1s, grad_logZ2s)
    if 'pdr' in test_types:
        diff_map['pdr'] = comp_mat_diff_pdr(n, θ1s, θ2s)
    if 'chalm' in test_types:
        diff_map['chalm'] = comp_mat_diff_chalm(n, θ1s, θ2s)
    if 'mcr' in test_types:
        diff_map['mcr'] = comp_mat_diff_mcr(n, θ1s, θ2s, grad_logZ1s, grad_logZ2s)
    
    if 'pdm' in test_types:
        # PDM 直接计算统计量，不经过差分数组
        tpdm_stat = comp_mat_stat_pdm(n, θ1s, θ2s, grad_logZ1s, grad_logZ2s)
        results['pdm'] = (tpdm_stat, np.nan)

    # 3. 置换检验设置
    js, exact = _mat_permutation_plan(len(θ1s), Lmax)

    # 4. 计算 P 值
    for t in test_types:
        if t == 'pdm': continue

        diffs = diff_map[t]
        perms = comp_mat_perm_stats(diffs, js)
        
        # 计算差异统计量（均值）
        stat_val = sum(diffs) / len(diffs)

        pval = _permutation_pvalue(stat_val, perms, exact)
        
        results[t] = (stat_val, float(pval))

    return results

# #check
# n = np.array([10])
# θ1s = np.array([[0.5, 0.5] for _ in range(5)])
# θ2s = np.array([[-0.5, 0.5] for _ in range(5)])
# print(mat_tests(n, θ1s, θ2s,test_types=['pdr', 'mcr']))


