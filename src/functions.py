import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import Inference as inf
import Bioinformatics as bio
import os
import pandas as pd
import warnings
import numpy as np
from typing import List
import time
warnings.filterwarnings("ignore")

def can_merge(a: List[int], b: List[int]) -> bool:
    """Determine whether two samples can be merged without overlapping observed positions."""
    for ai, bi in zip(a, b):
        if ai != -1 and bi != -1: 
            return False
    return True

def merge_samples(a: List[int], b: List[int]) -> List[int]:
    """Overlay two samples to produce a merged observation vector."""
    return [a[i] if a[i] != -1 else b[i] for i in range(len(a))]

def compress_xobs(xobs: List[List[int]]) -> List[List[int]]:
    """Compress xobs sample set by merging only non-overlapping observation vectors."""
    compressed = []
    for sample in xobs:
        merged = False
        for i, target in enumerate(compressed):
            if can_merge(target, sample):
                compressed[i] = merge_samples(target, sample)
                merged = True
                break
        if not merged:
            compressed.append(sample)
    return compressed


def visualize_xobs(xobs: List[List[int]], title: str):
    """Visualize the xobs matrix as a heatmap."""
    num_samples = len(xobs)
    length = len(xobs[0]) if num_samples > 0 else 0

    # Configure per-sample height and overall figure height
    sample_height = 0.01  # per-sample height in inches
    fig_height = max(4, num_samples * sample_height)

    fig, ax = plt.subplots(figsize=(12, fig_height))

    cmap = ListedColormap(['white', '#ff9cb0', '#8cf7ff'])

    # Prepare data by mapping -1, 0, 1 to 0, 1, 2
    data = np.array(xobs)
    data = np.where(data == -1, 0, data + 1)

    # Render the image while preserving equal physical height per sample
    cax = ax.imshow(data, cmap=cmap, aspect='auto', interpolation='none')

    # Set y-axis ticks at the sample centers
    ax.set_yticks([])

    # Set axis labels
    ax.set_xlabel('CpG-Sites')
    ax.set_title(title)

    # Add legend
    legend_elements = [
        patches.Patch(facecolor='#8cf7ff', label='1'),
        patches.Patch(facecolor='#ff9cb0', label='0'),
        patches.Patch(facecolor='white', label='-1')
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

## Compute n_list and theta_list for a single input file
def _cal_cpel(cpg_sites_file, data, chr, start, end, step=500, compress=False,
              vis=False, *, filter_coverage=False, min_cov=5.0, max_zero_frac=0.2):
    """Estimate one region; optionally require coverage >= min_cov and zero_frac <= limit.

    Returns (CpG counts, theta, metrics). Skipped regions return theta=[] and
    metrics['skip_reason']. The coverage denominator includes every CpG in the
    region, including uncovered sites. Empty genomic windows have no parameter.
    Filtering defaults to off for compatibility with existing callers.
    """
    if step <= 0 or end < start:
        raise ValueError("step must be positive and end must be >= start")
    if not np.isfinite(min_cov) or min_cov < 0:
        raise ValueError("min_cov must be finite and nonnegative")
    if not np.isfinite(max_zero_frac) or not 0 <= max_zero_frac <= 1:
        raise ValueError("max_zero_frac must lie in [0, 1]")

    cpg_n_list =[]  # CpG count for each segment
    xobs = []  # observation matrix
    # new_xobs = []  # observation matrix used for downstream estimation


    # Segment the interval and count CpGs per window
    for current_start in range(start, end + 1, step):
        current_end = min(current_start + step - 1, end)
        # print(f"[{current_start}, {current_end}]")
        # Retrieve CpG positions within the current genomic segment
        cpg_in_range = bio.get_cpg_in_range(chr, current_start, current_end, cpg_sites_file)
        # print(f"Number of CpGs in [{current_start}, {current_end}]:", len(cpg_in_range))
        # CpG count for this segment
        cnt = len(cpg_in_range)
        if cnt > 0:            # keep only segments with >0 CpGs
            cpg_n_list.append(cnt)

    total_cpg_total = sum(cpg_n_list)
    metrics = {
        "mean_cov": 0.0,
        "zero_cpg": total_cpg_total,
        "total_cpg": total_cpg_total,
        "zero_frac": (1.0 if total_cpg_total > 0 else float("nan")),
        "skipped": True,
        "skip_reason": "no_cpg" if not total_cpg_total else "no_reads",
    }
    if not total_cpg_total:
        return cpg_n_list, [], metrics


    # Extract reads overlapping the full target interval from the raw data
    result_df,cover_start, cover_end = bio.extract_reads_in_range(chr ,start ,end ,data)
    

    if result_df is None or result_df.empty:
        # No reads found; return empty theta
        theta = []
        return cpg_n_list, theta, metrics

    if cover_start is None or cover_end is None:
        cover_start, cover_end = start, end

    cover_start = min(start, cover_start)
    cover_end   = max(end, cover_end)
    cpg_all = bio.get_cpg_in_range(chr, cover_start, cover_end, cpg_sites_file)
    
    # xobs aligned to the maximal coverage region
    xobs_superset = bio.build_xobs_superset(result_df, cpg_all)

    # xobs restricted to the target ROI
    xobs=bio.build_xobs(xobs_superset, cpg_all, start, end )
    
    # Remove invalid rows comprised entirely of -1
    
    xobs_arr = np.array(xobs, dtype=int)
    if xobs_arr.size == 0:
        return cpg_n_list, [], metrics
    if xobs_arr.ndim != 2 or xobs_arr.shape[1] != total_cpg_total:
        raise ValueError("xobs column count does not match the region's CpG counts")
    new_xobs = np.zeros_like(xobs_arr) # default value 0 corresponds to -1
    new_xobs[xobs_arr == 0] = -1      # map 0 to -1
    new_xobs[xobs_arr == 1] = 1   
    
    
        # Keep rows that are not entirely zero
    new_xobs= new_xobs[~(new_xobs == 0).all(axis=1)]
    if len(new_xobs) == 0:
        metrics["skip_reason"] = "no_observed_cpg"
        return cpg_n_list, [], metrics
    


    if new_xobs.size > 0 and sum(cpg_n_list) != new_xobs.shape[1]:
        raise ValueError(
            f"shape mismatch: sum(n)={sum(cpg_n_list)} vs xobs width={new_xobs.shape[1]}. "
            "xobs column count does not match n."
    )
     ## compress and vis still use the original xobs

            # rows = reads, columns = CpGs
    mean_cov = float((np.abs(new_xobs) > 0).sum()) / new_xobs.shape[1]
    obs_per_cpg = (np.abs(new_xobs) > 0).sum(axis=0)
    zero_cpg = int((obs_per_cpg == 0).sum())
    total_cpg = new_xobs.shape[1]
    zero_frac = zero_cpg / total_cpg if total_cpg > 0 else float("nan")

    metrics = {
        "mean_cov": mean_cov,
        "zero_cpg": zero_cpg,
        "total_cpg": total_cpg,
        "zero_frac": zero_frac,
        "skipped": False,
        "skip_reason": None,
    }

    if filter_coverage:
        reasons = []
        if mean_cov < min_cov:
            reasons.append("mean_cov_below_min")
        if zero_frac > max_zero_frac:
            reasons.append("zero_frac_above_max")
        if reasons:
            metrics.update(skipped=True, skip_reason=";".join(reasons))
            return cpg_n_list, [], metrics

    if compress==True:
        xobs = compress_xobs(xobs)
    cpg_n = np.array(cpg_n_list, dtype=int)
    ##计算theta？？xbox没有标准化-1→0,1→1,0→-1 ,改为new_xobs 
    theta = inf.est_theta_sa(cpg_n ,new_xobs)


    if vis == True and compress==True:
        print("compressed_xobs:")
        visualize_xobs(xobs, f"Compressed xobs Matrix (Samples: {len(xobs)})")

    return cpg_n_list ,theta, metrics



def process_files(folder_path, chr, start, end, step, compress=False, vis=False,
                  *, cpg_sites_file=None, filter_coverage=False, min_cov=5.0,
                  max_zero_frac=0.2):
    """
    Process all files in a folder and compute theta values.
    A skipped sample retains an empty list in theta_list to preserve file order.
    """
    if cpg_sites_file is None:
        cpg_sites_file = globals().get("cpg_sites_file")
    if cpg_sites_file is None:
        raise ValueError("Pass cpg_sites_file to process_files")
    theta_list = []
    n_list = None

    # Enumerate all input files in the folder

    file_list = [f for f in os.listdir(folder_path) if f.endswith('.mhap.gz')]
    print(file_list)

    for file in file_list:
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path, sep='\t', header=None, dtype={3: 'str'})
        current_n, current_theta, metrics = _cal_cpel(
            cpg_sites_file, data, chr=chr, start=start, end=end, step=step,
            compress=compress, vis=vis, filter_coverage=filter_coverage,
            min_cov=min_cov, max_zero_frac=max_zero_frac)

        # Ensure n_list is consistent across all files
        if n_list is None:
            n_list = current_n
        else:
            assert n_list == current_n, "n_list mismatch"
        print(file, current_theta)
        if metrics["skipped"]:
            print("Skipped:", metrics["skip_reason"])
        theta_list.append(current_theta)

    return n_list, theta_list
