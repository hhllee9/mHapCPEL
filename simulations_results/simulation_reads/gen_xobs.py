from scipy.special import expit
import numpy as np
import os
import time
import argparse
from Inference import comp_lkhd, comp_log_g

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
    
    x[0] = 1 if np.random.rand() < initial_prob else -1

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

        x[i] = 1 if np.random.rand() < p else -1
    return x

def gen_xobs_fixed_coverage(n, θ, coverage):
    """Generate reads with fixed sequencing coverage"""
    N = int(np.sum(n))
    xobs = np.empty((coverage, N), dtype=int)
    for i in range(coverage):
        xobs[i] = gen_x_mc(n, θ)
    return xobs


# Parse command-line arguments and generate raw observation matrix

def main():
    parser = argparse.ArgumentParser(description="CPEL raw observation data generator (.npy)")
    parser.add_argument("-n", "--sites", type=int, default=4, help="Number of CpG sites (N), default: 4")
    parser.add_argument("-a", "--alpha", type=float, required=True, help="Methylation propensity parameter (Alpha)")
    parser.add_argument("-b", "--beta", type=float, required=True, help="Correlation parameter (Beta)")
    parser.add_argument("-c", "--coverage", type=int, required=True, help="Sequencing coverage (reads per sample)")
    parser.add_argument("-i", "--iterations", type=int, default=1000, help="Number of simulation iterations, default: 1000")
    parser.add_argument("-o", "--outdir", type=str, default="raw_xobs", help="Output directory, default: raw_xobs")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed for reproducibility, default: 42")
    args = parser.parse_args()
    
    np.random.seed(args.seed)

    print("====================================")
    print("Initializing CPEL raw xobs data generation")
    print(f"Parameters: N={args.sites}, Alpha={args.alpha}, Beta={args.beta}, Coverage={args.coverage}")
    print(f"Iterations: {args.iterations}")
    print("====================================")
    
    start_time = time.time()
    
    n = np.array([args.sites])
    θ = np.array([args.alpha, args.beta])

    # Pre-allocate a large 3D matrix to store all simulation results
    # Shape: (Iterations, Coverage, CpG sites)
    all_xobs = np.empty((args.iterations, args.coverage, args.sites), dtype=int)
    
    print("> Performing exact Monte Carlo sampling...")
    for i in range(args.iterations):
        all_xobs[i] = gen_xobs_fixed_coverage(n, θ, args.coverage)
    
   # Save the output matrix as a .npy file
    os.makedirs(args.outdir, exist_ok=True)
    filename = f"xobs_raw_N{args.sites}_a{args.alpha}_b{args.beta}_c{args.coverage}.npy"
    filepath = os.path.join(args.outdir, filename)
    
    np.save(filepath, all_xobs)
    
    end_time = time.time()
    print(f"\n> Simulation completed successfully!")
    print(f"> Data matrix shape: {all_xobs.shape}")
    print(f"> Raw data saved to: {filepath}")
    print(f"> Time elapsed: {(end_time - start_time):.2f} seconds")


if __name__ == "__main__":
    main()