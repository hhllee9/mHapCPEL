"""Four-CpG CPEL simulations with exact, nested end-missingness.

Example: python src/Simulations_missing.py unmat 5 --theta1=0.5,0.5 --theta2=0.5,0.5
Each iteration generates two independent groups, then reuses the complete data
at 0/25/50/75 percent missingness. Matched mode preserves the original simulator's
independent-sample generation and uses its paired test (no subject random effect).
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import scipy

from Simulations import gen_xobs, simulation_param_string
from Inference import create_Llkhd, est_theta_sa, ETA_MAX_ABS
from HypothesisTesting import mat_tests, unmat_tests

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / 'simulations_results/missing_data'
METHODS = ['MML', 'NME', 'PDR', 'CHALM', 'MCR']
NUMERIC_ERRORS = (ValueError, RuntimeError, FloatingPointError, OverflowError)


def seed_stream(seed, iteration, stage, *indices):
    """Independent streams for generation, masking, fitting and permutation tests."""
    return int(np.random.SeedSequence([seed, iteration, stage, *indices])
               .generate_state(1)[0])


def seed_globals(seed):
    # The existing sampler, optimizer and permutation tests use both globals.
    np.random.seed(seed)
    random.seed(seed)


def nested_end_masking(xobs, missing_percent, rng):
    """Mask exact counts per sample, trimming either end symmetrically.

    Last two axes are reads and CpGs. Every read retains a nonempty contiguous
    interval. Choose an eligible read uniformly, then its left/right end with
    equal probability; methylation values never influence this choice.
    Returned axes: missingness, <original axes>. Inputs are never modified.
    """
    xobs = np.asarray(xobs)
    if xobs.ndim < 2 or not all(xobs.shape) or not np.all(np.isin(xobs, [-1, 1])):
        raise ValueError('Expected nonempty complete reads with values -1/1')
    rates = list(missing_percent)
    if not rates or rates != sorted(set(rates)):
        raise ValueError('Missing percentages must be distinct and increasing')
    reads, sites = xobs.shape[-2:]
    targets = []
    for rate in rates:
        count = reads * sites * rate / 100
        if not np.isfinite(count) or rate < 0 or not np.isclose(count, round(count)):
            raise ValueError('Missingness must specify an integer number of entries per sample')
        count = int(round(count))
        if count > reads * (sites - 1):
            raise ValueError('Missingness would require empty reads')
        targets.append(count)
    output = np.empty((len(rates),) + xobs.shape, dtype=xobs.dtype)
    for index in np.ndindex(xobs.shape[:-2]):
        # A sample-local stream keeps other samples' masks unchanged when the
        # requested maximum missingness changes (e.g. omit the 75% condition).
        sample_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
        sample = xobs[index].copy()
        left = np.zeros(reads, dtype=int)
        right = np.full(reads, sites, dtype=int)
        dropped = 0
        for level, target in enumerate(targets):
            while dropped < target:
                eligible = np.flatnonzero(right - left > 1)
                row = int(sample_rng.choice(eligible))
                if sample_rng.integers(2) == 0:
                    sample[row, left[row]] = 0
                    left[row] += 1
                else:
                    right[row] -= 1
                    sample[row, right[row]] = 0
                dropped += 1
            output[(level,) + index] = sample
    return output


def parse_theta(value):
    try:
        theta = [float(part) for part in value.split(',')]
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Use alpha,beta, e.g. -1,1') from exc
    if len(theta) != 2 or not np.isfinite(theta).all():
        raise argparse.ArgumentTypeError('Exactly two finite values alpha,beta are required')
    return theta


def run_simulation(args):
    n = np.array([4])
    rates = args.missing_percent
    methods = METHODS.copy()
    # A new directory per invocation protects all previous complete/missing runs.
    tag = (f'{args.mode}_{simulation_param_string(args.samples, n, args.theta1, args.theta2)}'
           f'_c20_seed{args.seed}_{datetime.now():%Y%m%d_%H%M%S_%f}')
    outdir = args.output_dir.resolve() / tag
    outdir.mkdir(parents=True, exist_ok=False)
    metadata = {
        'mode': args.mode, 'samples_per_group': args.samples, 'n': [4],
        'theta1': args.theta1, 'theta2': args.theta2, 'reads_per_sample': 20,
        'missing_percent': rates, 'iterations_requested': args.iterations,
        'seed': args.seed, 'permutations': args.permutations, 'methods': methods,
        'checkpoint_every': args.checkpoint_every,
        'significance_level': args.alpha, 'status': 'running', 'completed_iterations': 0,
        'masking': 'nested symmetric end trimming; exact count per sample; at least one CpG per read',
        'stress_condition': 'At 75% missingness all reads contain exactly one observed CpG.',
        'matched_generation': 'Independent groups, as in Simulations.py; no subject random effect.',
        'failure_policy': 'Fixed attempts; no replacement of failed iterations or samples.',
        'optimization': 'Existing est_theta_sa; finite output is not a convergence certificate.',
        'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
        'sources_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                           for name in ['Simulations_missing.py', 'Simulations.py',
                                        'Inference.py', 'HypothesisTesting.py']},
    }
    meta_path = outdir / 'metadata.json'
    meta_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    reps, levels, m = args.iterations, len(rates), args.samples
    complete = np.zeros((reps, 2, m, 20, 4), dtype=np.int8)
    masked = np.zeros((levels, reps, 2, m, 20, 4), dtype=np.int8)
    estimates = np.full((levels, reps, 2, m, 2), np.nan)
    pvalues = np.full((levels, reps, len(methods)), np.nan)
    statistics = np.full_like(pvalues, np.nan)

    def checkpoint(completed, status):
        temporary = outdir / 'results.tmp.npz'
        np.savez_compressed(temporary, complete_xobs=complete[:completed],
                            masked_xobs=masked[:, :completed], theta_hat=estimates[:, :completed],
                            pvalues=pvalues[:, :completed], statistics=statistics[:, :completed],
                            missing_percent=np.array(rates), methods=np.array(methods))
        temporary.replace(outdir / 'results.npz')
        metadata.update(completed_iterations=completed, status=status)
        meta_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    print(f'Output: {outdir}', flush=True)
    if 75 in rates:
        print('75% is a stress condition: one observed CpG per read.', flush=True)
    with (outdir / 'failures.csv').open('w', newline='', encoding='utf-8') as error_file, \
         (outdir / 'sample_diagnostics.csv').open('w', newline='', encoding='utf-8') as diag_file:
        errors = csv.writer(error_file)
        errors.writerow(['iteration', 'missing_percent', 'stage', 'group', 'sample', 'message'])
        diagnostics = csv.writer(diag_file)
        diagnostics.writerow(['iteration', 'missing_percent', 'group', 'sample', 'actual_missing_percent',
                              'mean_coverage', 'cov_1', 'cov_2', 'cov_3', 'cov_4',
                              'adjacent_joint_observations', 'no_joint_observations',
                              'fit_at_boundary', 'negative_loglikelihood'])
        for iteration in range(reps):
            seed_globals(seed_stream(args.seed, iteration, 0))
            for group, theta in enumerate([args.theta1, args.theta2]):
                for sample in range(m):
                    complete[iteration, group, sample] = gen_xobs(n, np.array(theta), reads=20)
            masked[:, iteration] = nested_end_masking(
                complete[iteration], rates,
                np.random.default_rng(seed_stream(args.seed, iteration, 1)))
            for level, rate in enumerate(rates):
                valid_group = True
                for group in range(2):
                    for sample in range(m):
                        xobs = masked[level, iteration, group, sample]
                        observed = xobs != 0
                        coverage = observed.sum(axis=0)
                        joints = int(np.sum(observed[:, :-1] & observed[:, 1:]))
                        boundary, objective = '', np.nan
                        try:
                            # Same optimization stream across missing levels; separate from data/masks.
                            seed_globals(seed_stream(args.seed, iteration, 2, group, sample))
                            theta = np.asarray(est_theta_sa(n, xobs), dtype=float)
                            if theta.shape != (2,) or not np.isfinite(theta).all():
                                raise ValueError('Nonfinite or incorrectly sized parameter estimate')
                            objective = float(create_Llkhd(n, xobs)(theta))
                            if not np.isfinite(objective):
                                raise ValueError('Nonfinite fitted likelihood')
                            estimates[level, iteration, group, sample] = theta
                            boundary = bool(np.any(np.isclose(np.abs(theta), ETA_MAX_ABS, atol=1e-5)))
                        except NUMERIC_ERRORS as exc:
                            valid_group = False
                            errors.writerow([iteration + 1, rate, 'fit', group + 1, sample + 1,
                                             f'{type(exc).__name__}: {exc}'])
                        diagnostics.writerow([iteration + 1, rate, group + 1, sample + 1,
                                              100 * (1 - observed.mean()), coverage.mean(), *coverage,
                                              joints, joints == 0, boundary, objective])
                if not valid_group:
                    continue  # Keep NaN for this region/rate, without dropping or replacing samples.
                try:
                    seed_globals(seed_stream(args.seed, iteration, 3, rate))
                    test = mat_tests if args.mode == 'mat' else unmat_tests
                    result = test(n, estimates[level, iteration, 0], estimates[level, iteration, 1],
                                  Lmax=args.permutations, test_types=[name.lower() for name in methods])
                    for k, name in enumerate(methods):
                        stat, pval = result[name.lower()]
                        if not np.isfinite([stat, pval]).all() or not 0 <= pval <= 1:
                            errors.writerow([iteration + 1, rate, f'test:{name}', '', '',
                                             'Nonfinite statistic or invalid p-value'])
                            continue
                        statistics[level, iteration, k] = stat
                        pvalues[level, iteration, k] = pval
                except NUMERIC_ERRORS as exc:
                    errors.writerow([iteration + 1, rate, 'test', '', '', f'{type(exc).__name__}: {exc}'])
            error_file.flush()
            diag_file.flush()
            if (iteration + 1) % args.checkpoint_every == 0 or iteration + 1 == reps:
                checkpoint(iteration + 1, 'complete' if iteration + 1 == reps else 'running')
                print(f'Completed {iteration + 1}/{reps} region attempts', flush=True)
    write_summary(outdir, metadata, pvalues)
    return outdir


def write_summary(outdir, metadata, pvalues):
    with (outdir / 'summary.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['missing_percent', 'method', 'attempted', 'valid', 'failed',
                         'rejected', 'rejection_rate_valid', 'mcse_valid'])
        for level, rate in enumerate(metadata['missing_percent']):
            for k, method in enumerate(metadata['methods']):
                values = pvalues[level, :, k]
                finite = values[np.isfinite(values)]
                valid, attempted = len(finite), len(values)
                rejected = int(np.sum(finite <= metadata['significance_level']))
                proportion = rejected / valid if valid else np.nan
                mcse = np.sqrt(proportion * (1 - proportion) / valid) if valid else np.nan
                writer.writerow([rate, method, attempted, valid, attempted - valid,
                                 rejected, proportion, mcse])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['unmat', 'mat'])
    parser.add_argument('samples', type=int, help='Samples per group, or number of pairs')
    parser.add_argument('--theta1', type=parse_theta, required=True, help='Use --theta1=-1,1 for negative alpha')
    parser.add_argument('--theta2', type=parse_theta, required=True)
    parser.add_argument('--iterations', type=int, default=1000,
                        help='Number of newly generated region experiments (default: 1000)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--missing-percent', type=int, nargs='+', default=[0, 25, 50, 75])
    parser.add_argument('--permutations', type=int, default=1000,
                        help='Permutation budget within each test; enumerate all if fewer (default: 1000)')
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--checkpoint-every', type=int, default=10,
                        help='Save all completed experiments every N iterations, also at the end; '
                             'does not change the experiment count or enable automatic resume (default: 10)')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if min(args.samples, args.iterations, args.permutations, args.checkpoint_every) < 1:
        parser.error('Samples, iterations, permutations and checkpoint interval must be positive')
    if not 0 <= args.seed < 2**32 or not 0 < args.alpha < 1:
        parser.error('Seed must be in [0,2**32); alpha must be in (0,1)')
    if (args.missing_percent != sorted(set(args.missing_percent))
            or any(rate not in [0, 25, 50, 75] for rate in args.missing_percent)):
        parser.error('Missing percentages must be distinct, increasing members of 0 25 50 75')
    run_simulation(args)


if __name__ == '__main__':
    main()
