import numpy as np
from Inference import *
from HypothesisTesting import *
from scipy.stats import rv_discrete
from scipy.special import expit
from numpy.random import choice 
import sys
import os

def gen_x_mc(n, θ):
    alpha = θ[:-1]
    beta = float(θ[-1])

    total_len = int(np.sum(n))
    subid = np.repeat(np.arange(1, len(n) + 1, dtype=int), n)

    x = np.empty(total_len, dtype=int)

    initial_prob = float(
        comp_lkhd(
            np.insert(np.zeros(total_len - 1, dtype=int), 0, 1),
            n, alpha, beta
        )
    )
    x[0] = choice([-1, 1], p=[1 - initial_prob, initial_prob])

    if total_len <= 1:
        return x 

    for i in range(1, total_len):
        log_odds = 2.0 * (alpha[subid[i] - 1] + beta * x[i - 1])

        if i < total_len - 1:
            ap1p = 2.0 * beta + alpha[subid[i + 1] - 1]
            ap1m = -2.0 * beta + alpha[subid[i + 1] - 1]
            uniq, counts = np.unique(subid[i + 1:], return_counts=True)
            n_miss = counts.astype(int)
            log_gp = float(comp_log_g(n_miss, alpha[uniq - 1], beta, ap1p, alpha[-1]))
            log_gm = float(comp_log_g(n_miss, alpha[uniq - 1], beta, ap1m, alpha[-1]))
            p = float(expit(log_odds + log_gp - log_gm))
        else:
            p = float(expit(log_odds))

       # x[i] = choice([-1, 1], p=[1 - p, p])
        x[i] = 1 if np.random.rand() < p else -1

    return x  # np.ndarray


# n=np.array([5])
# θ=np.array([5.0,5.0])    
# xobs = [gen_x_mc(n, θ) for _ in range(5)] 
# # xobs_arr = np.array(xobs, dtype=int) 
# # es_theta=est_theta_sa(n,xobs_arr)
# # es_theta1=est_theta_sa1(n,xobs_arr)
# print(xobs) 


def gen_xobs(n: np.ndarray,
             θ: np.ndarray,
             reads: int | None = None,
             reads_range: tuple[int, int] = (10, 31)):

    if reads is None:
        reads = np.random.randint(reads_range[0], reads_range[1])  

    N = int(np.sum(n))

    xobs = np.empty((reads, N), dtype=int)
    for i in range(reads):
        xobs[i] = gen_x_mc(n, θ)
    return xobs

# n=np.array([5])
# θ=np.array([5.0,5.0])  
# print(gen_xobs(n,θ))
#print(gen_xobs(n, θ, reads=20))
#print(gen_xobs(n, θ, reads_range=(5, 11)))



## Comparing differences in data generated with different parameters
# m: Sample size for the two groups
# n: Number of CpG sites
# θ1: Original parameters for the first group
def unmatched_sim(m: int, n: np.ndarray, θ1: np.ndarray, θ2: np.ndarray):
    stat_keys = ['mml', 'nme', 'pdm', 'pdr', 'chalm', 'mcr']
    #Initialize storage structure：{ 'mml': [], 'nme': [], ... }
    storage = {k: [] for k in stat_keys}
    # tmml, tnme, tpdm, tpdr, tchalm, tmcr = [], [], [], [], [], []
    c = 1

    while len(storage['mml']) < 1000:
        # Generate xobs for m groups (with 10–30 reads per group, randomly determined) and estimate θ.
        g1_θs = [est_theta_sa(n, gen_xobs(n, θ1)) for _ in range(m)]
        g2_θs = [est_theta_sa(n, gen_xobs(n, θ2)) for _ in range(m)]
        test_dict = unmat_tests(n, np.asarray(g1_θs), np.asarray(g2_θs))

        for k in stat_keys:
            if k in test_dict:
                storage[k].append(test_dict[k])
    
        c += 1

    # return tmml, tnme, tpdm, tpdr, tchalm, tmcr
    return storage


def matched_sim(m, n, θ1, θ2):
    stat_keys = ['mml', 'nme', 'pdr', 'chalm', 'mcr']
    storage = {k: [] for k in stat_keys}
    # tmml, tnme, tpdm, tpdr, tchalm, tmcr = [], [], [], [], [], []
    c = 1

    # tmml, tnme, tpdm, tpdr, tchalm, tmcr = [], [], [], [], [], []
    # c = 1

    while len(storage['mml']) < 1000:
        # Per group: 10–30 randomly selected reads
        g1_θs = [est_theta_sa(n, gen_xobs(n, θ1)) for _ in range(m)]
        g2_θs = [est_theta_sa(n, gen_xobs(n, θ2)) for _ in range(m)]
        test_dict = mat_tests(n, np.asarray(g1_θs), np.asarray(g2_θs))
         
        for k in stat_keys:
            if k in test_dict:
                storage[k].append(test_dict[k])
    

        c += 1

    # return tmml, tnme, tpdm, tpdr, tchalm, tmcr
    return storage

 
    
# Empirical estimate of the cumulative distribution function of the p-value for the difference test statistic (mismatched)
#m: Number of samples per group
def unmat_pvalue_ecdf(m: int, n: np.ndarray, θ1: np.ndarray, θ2:  np.ndarray):
    
    sim_results = unmatched_sim(m, n, θ1, θ2)
  
    pvals_dict = {k.upper(): [res[1] for res in v] for k, v in sim_results.items()}
    
  
    ecdf_funcs = {}
    for stat_name, pvals in pvals_dict.items():
        sorted_pvals = np.sort(pvals)

        ecdf_funcs[stat_name] = lambda x, data=sorted_pvals: np.searchsorted(data, x, side='right') / len(data)

    return ecdf_funcs, pvals_dict





def mat_pvalue_ecdf(m: int, n: np.ndarray, θ1: np.ndarray, θ2: np.ndarray):
    """
    Paired comparison: Extract p-values ​​and construct the empirical cumulative distribution function (ECDF).
    """
    sim_results = matched_sim(m, n, θ1, θ2)
    pvals_dict = {k.upper(): [res[1] for res in v] for k, v in sim_results.items()}
    ecdf_funcs = {}
    for stat_name, pvals in pvals_dict.items():
        sorted_pvals = np.sort(pvals)
        ecdf_funcs[stat_name] = lambda x, data=sorted_pvals: np.searchsorted(data, x, side='right') / len(data)
    
    return ecdf_funcs, pvals_dict



if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("usage: python Simulations_new.py <unmat/mat> <m> <n> <theta1> <theta2>")
        print("Example: python Simulations_new.py unmat 5 4 0.5,0.5 0.5,0.5")
        sys.exit(1)

    func_name = sys.argv[1]
    m = int(sys.argv[2])
    n = np.fromstring(sys.argv[3], sep=',', dtype=int).reshape(-1)
    θ1 = np.fromstring(sys.argv[4], sep=',', dtype=float).reshape(-1)
    θ2 = np.fromstring(sys.argv[5], sep=',', dtype=float).reshape(-1)
    output_dir = "simulation_data"
    os.makedirs(output_dir, exist_ok=True)
    param_str = f"m{m}_n{n[0]}_theta1_{θ1[0]}_{θ1[1]}_theta2_{θ2[0]}_{θ2[1]}"
    if func_name == "unmat":
        print(f"Starting unpaired simulation...: m={m}, n={n}, θ1={θ1}, θ2={θ2}")
        ecdf_funcs, pvals_dict = unmat_pvalue_ecdf(m, n, θ1, θ2)
        save_path = f"{output_dir}/unmat_{param_str}_pvals.npy"
        np.save(save_path, pvals_dict, allow_pickle=True)
        print(f"Results saved to: {save_path}")

    elif func_name == "mat":
        print(f"Starting paired simulation...: m={m}, n={n}, θ1={θ1}, θ2={θ2}")
        ecdf_funcs, pvals_dict = mat_pvalue_ecdf(m, n, θ1, θ2)
        
        save_path = f"{output_dir}/mat_{param_str}_pvals.npy"
        np.save(save_path, pvals_dict, allow_pickle=True)
        print(f"Results saved to: {save_path}")
    
    else:
        print(f"Error: Unknown function name '{func_name}'. Please use 'unmat' or 'mat'.")

def comp_stat(n: np.ndarray, xobs: np.ndarray) -> List[float]:
    θ = est_theta_sa(n, xobs)
    grad_logZ = get_grad_logZ(n,θ)
    model_mml = comp_mml(n, grad_logZ)
    model_nme= comp_nme(n,θ,grad_logZ)
    model_pdr=comp_pdr(n, θ)
    model_chalm = comp_chalm(n, θ)
    model_mcr = model_chalm - model_mml

    total_reads = len(xobs)
    total_CpGs = xobs.size
    methy_CpGs = np.sum(xobs == 1)
    concordant_reads = np.all(xobs == xobs[:, [0]], axis=1)
    cpdr = np.sum(concordant_reads) / total_reads
    methy_read = np.any(xobs == 1, axis=1)
    methy_reads = xobs[methy_read]
    unmethy_GpGs = np.sum(methy_reads == -1)

    true_mml =np.round(methy_CpGs / total_CpGs,8).item()

    _, unique_reads_counts = np.unique(xobs, axis=0, return_counts=True)
    p = unique_reads_counts / total_reads
    with np.errstate(divide='ignore'):  
        log_p = np.log2(p)
    log_p[p == 0] = 0 
    true_nme =np.round(-1.0/sum(n)*np.sum(p * log_p),8).item()
    
    true_pdr = 1.0-np.round(cpdr,8).item()
    true_chalm = np.round(np.sum(methy_read) / total_reads,8).item()
    # true_mcr= unmethy_GpGs / total_CpGs
    # true_mcr_link = true_chalm-true_mml
    true_mcr = true_chalm-true_mml    

    return (model_mml, true_mml), (model_nme, true_nme), (model_pdr, true_pdr), (model_chalm, true_chalm), (model_mcr,  true_mcr)


