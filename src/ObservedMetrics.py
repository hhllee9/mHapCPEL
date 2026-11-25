"""Eleven regional metrics directly from read-by-CpG xobs (-1/0/+1).

Zero means missing, not unmethylated. No imputation or CPEL fitting is used.
PDR uses all input reads; CHALM uses nonempty reads; NME uses complete reads.
qFDRP pools valid paired CpG comparisons, so qFDRP = 1 - RC. The diagnostic
qFDRP_pair instead gives each eligible read pair equal weight.
MeConcord NRC/NCC subtract their direction-specific expected concordance;
diagnostic P_R/P_C use original ordered counts and are not group-test P-values.
Undefined quantities remain NaN.
"""
from __future__ import annotations

from math import nan
from typing import Any

import numpy as np
from scipy.stats import binom


METRIC_NAMES = (
    "MML", "NME", "PDR", "CHALM", "MCR", "FDRP",
    "qFDRP", "RC", "CC", "NRC", "NCC",
)


def _ratio(numerator: int | float, denominator: int | float) -> float:
    """Return NaN when the quantity has no eligible observations."""
    return float(numerator / denominator) if denominator else nan


def _binomial_tail(successes: int, total: int, expected: float) -> float:
    """Reproduce the selected tail in MeConcord equations 14 and 20."""
    if not total or not np.isfinite(expected):
        return nan
    if successes < total * expected:
        return float(binom.cdf(successes, total, expected))
    return float(binom.sf(successes - 1, total, expected))


def _read_pair_statistics(
    observations: np.ndarray, pair_block_size: int
) -> dict[str, int | float]:
    """Count all unordered pairs exactly using pattern multiplicities and blocks."""
    patterns, counts = np.unique(observations, axis=0, return_counts=True)
    signed = patterns.astype(np.int64)
    lengths = np.count_nonzero(signed, axis=1)
    within = counts * (counts - 1) // 2
    valid_within = lengths > 0
    n_valid_pairs = int(within[valid_within].sum())
    n_discordant_pairs = 0
    h_sum = int(np.sum(within * lengths))
    d_sum = 0
    ratio_sum = 0.0
    n_patterns = len(patterns)

    for first in range(0, n_patterns, pair_block_size):
        left = signed[first:first + pair_block_size]
        left_observed = np.abs(left)
        left_counts = counts[first:first + pair_block_size]
        for second in range(first, n_patterns, pair_block_size):
            right = signed[second:second + pair_block_size]
            shared = left_observed @ np.abs(right).T
            # Signed products are +1 for agreement, -1 for discordance and
            # zero for missing calls, so H - signed_product equals 2*D.
            discordant = (shared - left @ right.T) // 2
            eligible = shared > 0
            if first == second:
                eligible = np.triu(eligible, k=1)
            if not np.any(eligible):
                continue
            weights = left_counts[:, None] * counts[
                second:second + pair_block_size
            ][None, :]
            pair_weights = weights[eligible]
            pair_shared = shared[eligible]
            pair_discordant = discordant[eligible]
            n_valid_pairs += int(pair_weights.sum())
            n_discordant_pairs += int(pair_weights[pair_discordant > 0].sum())
            h_sum += int(np.sum(pair_weights * pair_shared))
            d_sum += int(np.sum(pair_weights * pair_discordant))
            ratio_sum += float(np.sum(pair_weights * (pair_discordant / pair_shared)))

    return {
        "n_unique_patterns": int(n_patterns),
        "n_valid_pairs": n_valid_pairs,
        "n_discordant_pairs": n_discordant_pairs,
        "Hsum": h_sum,
        "Dsum": d_sum,
        "qFDRP_pair": _ratio(ratio_sum, n_valid_pairs),
    }


def compute_region_metrics(
    xobs: Any, include_diagnostics: bool = False, *, pair_block_size: int = 256
) -> dict[str, float] | tuple[dict[str, float], dict[str, Any]]:
    """Return metrics, or (metrics, diagnostic counts) when requested.

    Input has shape (reads, CpGs), with at least one CpG; it is never modified.
    Zero denominators and absence of complete reads produce NaN. All-zero rows
    still enter the PDR denominator. pair_block_size limits pair-count memory
    without sampling or changing weights.
    """
    if (
        isinstance(pair_block_size, (bool, np.bool_))
        or not isinstance(pair_block_size, (int, np.integer))
        or pair_block_size < 1
    ):
        raise ValueError("pair_block_size must be a positive integer")
    pair_block_size = int(pair_block_size)
    observations = np.asarray(xobs)
    if observations.ndim != 2 or observations.shape[1] == 0:
        raise ValueError("xobs must have shape (reads, CpGs), with at least one CpG")
    if observations.dtype.kind not in "biuf" or not np.isin(observations, (-1, 0, 1)).all():
        raise ValueError("xobs values must be -1, 0 (missing), or +1")
    n_reads, n_cpgs = observations.shape
    observed = (observations != 0).astype(np.int64)
    methylated = (observations == 1).astype(np.int64)
    unmethylated = (observations == -1).astype(np.int64)

    read_lengths = observed.sum(axis=1)
    methylated_per_read = methylated.sum(axis=1)
    unmethylated_per_read = unmethylated.sum(axis=1)
    coverage = observed.sum(axis=0)
    methylated_per_cpg = methylated.sum(axis=0)
    unmethylated_per_cpg = unmethylated.sum(axis=0)

    n_observed = int(read_lengths.sum())
    n_methylated = int(methylated_per_read.sum())
    n_effective_reads = int(np.count_nonzero(read_lengths))
    mixed_reads = (methylated_per_read > 0) & (unmethylated_per_read > 0)
    n_discordant_reads = int(mixed_reads.sum())
    n_methylated_reads = int(np.count_nonzero(methylated_per_read))
    concurrence_calls = int(unmethylated_per_read[mixed_reads].sum())

    complete_mask = read_lengths == n_cpgs
    n_complete_reads = int(complete_mask.sum())
    if n_complete_reads:
        patterns, pattern_counts = np.unique(
            observations[complete_mask], axis=0, return_counts=True
        )
        probabilities = pattern_counts.astype(float) / n_complete_reads
        nme = float(np.sum(-probabilities * np.log2(probabilities)) / n_cpgs)
    else:
        patterns = np.empty((0, n_cpgs), dtype=np.int8)
        pattern_counts = np.empty(0, dtype=np.int64)
        nme = nan

    pair_counts = _read_pair_statistics(observations, pair_block_size)
    n_valid_pairs = pair_counts["n_valid_pairs"]
    n_discordant_pairs = pair_counts["n_discordant_pairs"]
    h_sum = pair_counts["Hsum"]
    d_sum = pair_counts["Dsum"]

    # Original MeConcord matrix sums use ordered off-diagonal comparisons.
    # Their ratios equal the unordered-pair ratios, but the original binomial
    # tail must keep these ordered totals rather than divide them by two.
    mr = int(np.sum(methylated_per_cpg * (methylated_per_cpg - 1)))
    nr = int(np.sum(unmethylated_per_cpg * (unmethylated_per_cpg - 1)))
    tr = int(np.sum(coverage * (coverage - 1)))
    mc = int(np.sum(methylated_per_read * (methylated_per_read - 1)))
    nc = int(np.sum(unmethylated_per_read * (unmethylated_per_read - 1)))
    tc = int(np.sum(read_lengths * (read_lengths - 1)))
    mpr = int(np.sum(methylated_per_cpg * (coverage - 1)))
    npr = int(np.sum(unmethylated_per_cpg * (coverage - 1)))
    mpc = int(np.sum(methylated_per_read * (read_lengths - 1)))
    npc = int(np.sum(unmethylated_per_read * (read_lengths - 1)))
    p_r = _ratio(mpr, mpr + npr)
    p_c = _ratio(mpc, mpc + npc)
    e_r = float(p_r ** 2 + (1 - p_r) ** 2)
    e_c = float(p_c ** 2 + (1 - p_c) ** 2)
    rc = _ratio(h_sum - d_sum, h_sum)
    cc = _ratio(mc + nc, tc)

    metrics = {
        "MML": _ratio(n_methylated, n_observed),
        "NME": nme,
        "PDR": _ratio(n_discordant_reads, n_reads),
        "CHALM": _ratio(n_methylated_reads, n_effective_reads),
        "MCR": _ratio(concurrence_calls, n_observed),
        "FDRP": _ratio(n_discordant_pairs, n_valid_pairs),
        "qFDRP": _ratio(d_sum, h_sum),
        "RC": rc,
        "CC": cc,
        "NRC": float(rc - e_r),
        "NCC": float(cc - e_c),
    }
    if not include_diagnostics:
        return metrics

    diagnostics = {
        "n_reads": int(n_reads),
        "n_unique_patterns": pair_counts["n_unique_patterns"],
        "n_cpgs": int(n_cpgs),
        "n_observed": n_observed,
        "n_methylated": n_methylated,
        "n_effective_reads": n_effective_reads,
        "n_complete_reads": n_complete_reads,
        "n_discordant_reads": n_discordant_reads,
        "n_methylated_reads": n_methylated_reads,
        "concurrence_calls": concurrence_calls,
        "coverage": coverage.tolist(),
        "read_lengths": read_lengths.tolist(),
        "methylated_per_cpg": methylated_per_cpg.tolist(),
        "methylated_per_read": methylated_per_read.tolist(),
        "complete_patterns": patterns.astype(np.int8).tolist(),
        "complete_pattern_counts": pattern_counts.tolist(),
        "n_total_pairs": int(n_reads * (n_reads - 1) // 2),
        "n_valid_pairs": n_valid_pairs,
        "n_discordant_pairs": n_discordant_pairs,
        "Hsum": h_sum,
        "Dsum": d_sum,
        "qFDRP_pair": pair_counts["qFDRP_pair"],
        "mr": mr, "nr": nr, "tr": tr,
        "mc": mc, "nc": nc, "tc": tc,
        "mpr": mpr, "npr": npr, "mpc": mpc, "npc": npc,
        "pR": p_r, "pC": p_c, "eR": e_r, "eC": e_c,
        "P_R": _binomial_tail(mr + nr, tr, e_r),
        "P_C": _binomial_tail(mc + nc, tc, e_c),
        "meconcord_counting": "ordered_off_diagonal_pairs",
    }
    return metrics, diagnostics
