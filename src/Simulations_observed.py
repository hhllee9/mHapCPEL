"""Regional observed metrics and the original sample-level permutation tests.

Reuse third-experiment reads:
python src/Simulations_observed.py unmat 5 --input <run>/results.npz

No CPEL parameter estimation is performed. Existing standalone data generation
is retained for small checks; formal experiment 4 reads the supplied archives.
"""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np

from HypothesisTesting import mat_observed_tests, unmat_observed_tests
from ObservedMetrics import METRIC_NAMES, compute_region_metrics
from Simulations import gen_xobs, simulation_param_string
from Simulations_missing import nested_end_masking, parse_theta, seed_globals, seed_stream, write_summary

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / 'simulations_results/simulations_observed'
READS, SITES = 20, 4
NUMERIC_ERRORS = (ValueError, RuntimeError, FloatingPointError, OverflowError)


def _json_safe(value):
    """Represent undefined diagnostics as JSON null."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return None if isinstance(value, float) and not np.isfinite(value) else value


def load_saved_xobs(path, samples, rates, iterations=None, mode=None):
    """Select existing xobs; never load fitted parameters or model P-values."""
    path = Path(path).resolve()
    if path.is_dir():
        path /= 'results.npz'
    with np.load(path, allow_pickle=False) as saved:
        complete = saved['complete_xobs']
        available_rates = saved['missing_percent'].tolist()
        selected = [available_rates.index(rate) for rate in rates]
        masked = saved['masked_xobs'][selected]
    available_iterations = len(complete)
    if complete.shape[1:] != (2, samples, READS, SITES):
        raise ValueError('Input shape disagrees with groups, samples, reads or CpGs')
    count = available_iterations if iterations is None else iterations
    if not 1 <= count <= available_iterations:
        raise ValueError('Requested iterations exceed the saved data or are not positive')
    metadata = json.loads(path.with_name('metadata.json').read_text(encoding='utf-8'))
    if mode is not None and metadata.get('mode') != mode:
        raise ValueError('Requested mode disagrees with the saved input mode')
    source = dict(path=str(path), metadata=metadata,
                  available_iterations=available_iterations, saved_rates=available_rates)
    return complete[:count], masked[:, :count], source


def run_simulation(args):
    rates, methods = list(args.missing_percent), list(METRIC_NAMES)
    source = None
    if args.input is not None:
        complete, masked, source = load_saved_xobs(
            args.input, args.samples, rates, args.iterations, args.mode)
        theta1, theta2 = source['metadata']['theta1'], source['metadata']['theta2']
        reps = len(complete)
    else:
        if args.theta1 is None or args.theta2 is None:
            raise ValueError('Generation requires theta1 and theta2')
        theta1, theta2 = args.theta1, args.theta2
        reps = 1000 if args.iterations is None else args.iterations
        complete = np.zeros((reps, 2, args.samples, READS, SITES), dtype=np.int8)
        masked = np.zeros((len(rates),) + complete.shape, dtype=np.int8)

    tag = (f'{args.mode}_{simulation_param_string(args.samples, np.array([SITES]), theta1, theta2)}'
           f'_seed{args.seed}_{datetime.now():%Y%m%d_%H%M%S_%f}')
    outdir = Path(args.output_dir).resolve() / tag
    outdir.mkdir(parents=True, exist_ok=False)
    metadata = dict(
        experiment='Direct observed regional metrics', mode=args.mode,
        samples_per_group=args.samples, n=[SITES], theta1=theta1, theta2=theta2,
        reads_per_sample=READS, missing_percent=rates, methods=methods, seed=args.seed,
        data_seed=args.seed if source is None else source['metadata'].get('seed'),
        permutations=args.permutations, significance_level=args.alpha,
        checkpoint_every=args.checkpoint_every, iterations_requested=reps,
        completed_iterations=0, status='running', parameter_estimation=False,
        input_source=source,
        test_engine='HypothesisTesting.unmat_observed_tests / mat_observed_tests',
        test_statistic='Sample group-1 mean minus group-2 mean; paired differences in mat mode',
        failure_policy='Fixed samples and attempts; an undefined sample invalidates only that metric',
        redundant_metric_policy='qFDRP=1-RC; opposite differences and identical two-sided P-values',
    )
    metric_values = np.full((len(rates), reps, 2, args.samples, len(methods)), np.nan)
    statistics = np.full((len(rates), reps, len(methods)), np.nan)
    pvalues = np.full_like(statistics, np.nan)

    def checkpoint(completed, status):
        temporary = outdir / 'results.tmp.npz'
        np.savez_compressed(
            temporary, complete_xobs=complete[:completed], masked_xobs=masked[:, :completed],
            sample_metrics=metric_values[:, :completed], statistics=statistics[:, :completed],
            pvalues=pvalues[:, :completed], missing_percent=np.array(rates), methods=np.array(methods))
        # Windows readers may briefly hold the previous checkpoint open.
        for attempt in range(5):
            try:
                temporary.replace(outdir / 'results.npz')
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(.2)
        metadata.update(completed_iterations=completed, status=status)
        (outdir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        write_summary(outdir, metadata, pvalues[:, :completed])

    print(f'Output: {outdir}', flush=True)
    checkpoint(0, 'running')
    completed = 0
    with (outdir / 'failures.csv').open('w', newline='', encoding='utf-8') as error_file, \
         (outdir / 'sample_metrics.csv').open('w', newline='', encoding='utf-8') as metric_file, \
         (outdir / 'sample_diagnostics.jsonl').open('w', encoding='utf-8') as diag_file:
        errors = csv.writer(error_file)
        errors.writerow(['iteration', 'missing_percent', 'stage', 'group', 'sample', 'message'])
        metric_writer = csv.writer(metric_file)
        metric_writer.writerow(['iteration', 'missing_percent', 'group', 'sample', 'pair_id', *methods])
        try:
            for iteration in range(reps):
                if source is None:
                    seed_globals(seed_stream(args.seed, iteration, 0))
                    for group, theta in enumerate((theta1, theta2)):
                        for sample in range(args.samples):
                            complete[iteration, group, sample] = gen_xobs(
                                np.array([SITES]), np.asarray(theta), reads=READS)
                    masked[:, iteration] = nested_end_masking(
                        complete[iteration], rates,
                        np.random.default_rng(seed_stream(args.seed, iteration, 1)))
                for level, rate in enumerate(rates):
                    for group in range(2):
                        for sample in range(args.samples):
                            try:
                                scores, diagnostics = compute_region_metrics(
                                    masked[level, iteration, group, sample], include_diagnostics=True)
                                values = [scores[name] for name in methods]
                                metric_values[level, iteration, group, sample] = values
                                for name, value in zip(methods, values):
                                    if not np.isfinite(value):
                                        errors.writerow([iteration + 1, rate, f'metric:{name}',
                                                         group + 1, sample + 1, 'Undefined sample metric'])
                                diag_file.write(json.dumps(_json_safe(dict(
                                    iteration=iteration + 1, missing_percent=rate,
                                    group=group + 1, sample=sample + 1,
                                    diagnostics=diagnostics)), allow_nan=False) + '\n')
                            except NUMERIC_ERRORS as exc:
                                errors.writerow([iteration + 1, rate, 'metrics', group + 1,
                                                 sample + 1, f'{type(exc).__name__}: {exc}'])
                            metric_writer.writerow([
                                iteration + 1, rate, group + 1, sample + 1,
                                sample + 1 if args.mode == 'mat' else '',
                                *metric_values[level, iteration, group, sample]])

                    seed_globals(seed_stream(args.seed, iteration, 3, int(rate)))
                    test = mat_observed_tests if args.mode == 'mat' else unmat_observed_tests
                    try:
                        result = test(metric_values[level, iteration, 0],
                                      metric_values[level, iteration, 1],
                                      Lmax=args.permutations, test_types=methods)
                        # These are complementary representations of the same read-pair score.
                        if np.isfinite(result['qFDRP']).all() and np.isfinite(result['RC']).all():
                            result['RC'] = (-result['qFDRP'][0], result['qFDRP'][1])
                        for k, name in enumerate(methods):
                            stat, pval = result[name]
                            if np.isfinite([stat, pval]).all() and 0 <= pval <= 1:
                                statistics[level, iteration, k], pvalues[level, iteration, k] = stat, pval
                            else:
                                errors.writerow([iteration + 1, rate, f'test:{name}', '', '',
                                                 'Undefined metric in a fixed sample or invalid test'])
                    except NUMERIC_ERRORS as exc:
                        errors.writerow([iteration + 1, rate, 'test', '', '', f'{type(exc).__name__}: {exc}'])
                completed = iteration + 1
                if completed % args.checkpoint_every == 0 or completed == reps:
                    for handle in (error_file, metric_file, diag_file):
                        handle.flush()
                    checkpoint(completed, 'complete' if completed == reps else 'running')
                    print(f'Completed {completed}/{reps} region attempts', flush=True)
        except KeyboardInterrupt:
            checkpoint(completed, 'interrupted')
            raise
        except Exception:
            checkpoint(completed, 'failed')
            raise
    return outdir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['unmat', 'mat'])
    parser.add_argument('samples', type=int)
    parser.add_argument('--input', type=Path, help='Third-experiment NPZ or its run directory')
    parser.add_argument('--theta1', type=parse_theta, help='Standalone generation only')
    parser.add_argument('--theta2', type=parse_theta, help='Standalone generation only')
    parser.add_argument('--iterations', type=int, help='Default: all saved repetitions, or 1000 when generating')
    parser.add_argument('--missing-percent', type=int, nargs='+', choices=[0, 25], default=[0, 25])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--permutations', type=int, default=1000)
    parser.add_argument('--alpha', type=float, default=.05)
    parser.add_argument('--checkpoint-every', type=int, default=10)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    run_simulation(parser.parse_args())


if __name__ == '__main__':
    main()
