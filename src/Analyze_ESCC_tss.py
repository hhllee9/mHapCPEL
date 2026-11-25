"""Fit remaining TSS regions and include reused parameters in each sample CSV."""
import csv
import hashlib
import os
from multiprocessing import Pool
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import functions as fn

PROJECT_DIR = Path('/sibcb1/bioinformatics/lihenghui/cpelmhap')
BED_RELATIVE = Path('ESCC_TSS/annotations/promoters_to_fit_1kb_closed.bed')
REUSED_RELATIVE = Path('ESCC_TSS/reused_parameters')
MASTER_SEED = 42
ALPHA_BP = 250
MIN_COVERAGE = 5
MAX_ZERO_FRACTION = 0.2
SAMPLES = (
    ('SRX8208802', 'normal', '1'),
    ('SRX8208803', 'normal', '2'),
    ('SRX8208804', 'normal', '6'),
    ('SRX8208812', 'tumor', '1'),
    ('SRX8208813', 'tumor', '2'),
    ('SRX8208814', 'tumor', '6'),
    ('SRX8208805', 'normal', '7'),
    ('SRX8208806', 'normal', '8'),
    ('SRX8208815', 'tumor', '7'),
    ('SRX8208816', 'tumor', '8'),
)
REGION_FIELDS = ('region_id', 'chrom', 'start', 'end')
RESULT_FIELDS = (*REGION_FIELDS, 'n_cpg_1', 'n_cpg_2', 'n_cpg_3', 'n_cpg_4',
                 'alpha_1', 'alpha_2', 'alpha_3', 'alpha_4', 'beta')


def read_regions(bed_file):
    regions = []
    with Path(bed_file).open(encoding='utf-8-sig') as handle:
        for line in handle:
            chrom, start, end, region_id = line.split()
            regions.append(dict(region_id=region_id, chrom=chrom, start=int(start), end=int(end)))
    return regions


def fit_seed(sample_id, region):
    label = f'{MASTER_SEED}|{sample_id}|{region["chrom"]}|{region["start"]}|{region["end"]}'
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], 'little')


def estimate_region(region, sample, data, cpg_file):
    row = {field: '' for field in RESULT_FIELDS}
    row.update(region)
    seed = fit_seed(sample[0], region)
    random.seed(seed)
    np.random.seed(seed)
    try:
        n, theta, qc = fn._cal_cpel(
            str(cpg_file), data, region['chrom'], region['start'], region['end'],
            step=ALPHA_BP, compress=False, vis=False,
            filter_coverage=True, min_cov=MIN_COVERAGE, max_zero_frac=MAX_ZERO_FRACTION,
        )
        row.update({f'n_cpg_{i+1}': int(count) for i, count in enumerate(n)})
        if not qc['skipped']:
            for column, alpha in zip(np.flatnonzero(n), theta[:-1]):
                row[f'alpha_{column+1}'] = float(alpha)
            row['beta'] = float(theta[-1])
    except Exception as exc:
        print(f'{sample[0]} {region["region_id"]}: {exc}', file=sys.stderr, flush=True)
    return row


def initialize_worker(sample, data, cpg_file):
    global _SAMPLE, _DATA, _CPG_FILE
    _SAMPLE, _DATA, _CPG_FILE = sample, data, cpg_file


def worker(region):
    return estimate_region(region, _SAMPLE, _DATA, _CPG_FILE)


def run_sample(sample, workers, job_id, project_dir=PROJECT_DIR):
    regions = read_regions(project_dir / BED_RELATIVE)
    reused = pd.read_csv(project_dir / REUSED_RELATIVE / sample[0] /
                         f'parameters_seed{MASTER_SEED}.csv', float_precision='round_trip').fillna('')
    # Read this sample once; Bioinformatics performs the only region filtering.
    data = pd.read_csv(project_dir / 'data' / f'{sample[0]}.mhap.gz',
                       sep='\t', header=None, dtype={0: str, 1: int, 2: int, 3: str, 4: int},
                       low_memory=False)
    cpg_file = project_dir / 'data/hg19_CpG.gz'
    outdir = project_dir / 'ESCC_TSS/results_escc_parameters' / f'job_{job_id}' / sample[0]
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / f'parameters_seed{MASTER_SEED}.csv').open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(reused.to_dict('records'))
        with Pool(workers, initializer=initialize_worker, initargs=(sample, data, cpg_file)) as pool:
            writer.writerows(pool.imap_unordered(worker, regions))
    return outdir


if __name__ == '__main__':
    sample = SAMPLES[int(os.environ['SGE_TASK_ID']) - 1]
    run_sample(sample, int(os.environ['NSLOTS']), os.environ['JOB_ID'])
