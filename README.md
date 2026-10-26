# mHapCPEL: A Python Package for DNA Methylation Haplotypes Analysis via the CPEL Model

## Introduction
mHapCPEL is a comprehensive Python toolkit designed for modeling DNA methylation haplotype data. This package builds upon the Julia package [CpelTdm.jl](https://github.com/jordiabante/CpelTdm.jl) and its underlying CPEL method.

Our objective goes beyond migrating the algorithm to Python. Based on the CPEL model, we have expanded the statistical metrics beyond the traditional Mean Methylation Level (MML) and Normalized Methylation Entropy (NME). In addition to performing differential analysis, this tool utilizes simulations to investigate the dynamic relationship between model parameters and these statistical metrics.

---

##  Requirements & Installation

* **Requirements:** Python 3.10+

Clone the repository and install the dependencies:

```bash
git clone [https://github.com/hhllee9/CPELmHap.git](https://github.com/hhllee9/CPELmHap.git)
cd CPELmHap

# Install dependencies
pip install numpy==1.23.4 pandas scipy pytabix statsmodels matplotlib
```
## Data Input
The core input file format is the mHap file, which can be generated using [mHapSuite](https://github.com/yoyoong/mHapSuite). Additionally, an annotation file containing CpG positions is required, which is also available via the [mHapSuite](https://github.com/yoyoong/mHapSuite) database.

|Data Type|Description|Format|Index|
| ------: | ---------: | :------------: | :------- | 
|Test File|Normal esophageal tissue (GSM4505856)|[SRX8208802.mhap.gz](http://bioinformatics.sibcb.ac.cn/dataupload/DataSets/ESCC/mHap/SRX8208802.mhap.gz) | [SRX8208802.mhap.gz.tbi](http://bioinformatics.sibcb.ac.cn/dataupload/DataSets/ESCC/mHap/SRX8208802.mhap.gz.tbi) |
|Annotation File|Whole genome hg19|[hg19_CpG.gz](http://bioinformatics.sibcb.ac.cn/dataupload/iGenome/CpGs/hg19/hg19_CpG.gz) | [hg19_CpG.gz.tbi](http://bioinformatics.sibcb.ac.cn/dataupload/iGenome/CpGs/hg19/hg19_CpG.gz.tbi) | 

## Parameter Estimation
You can use the test data to infer the parameters of the CPEL model. Based on the specified chromosomal region and a fixed sliding window size (default step=500), you can estimate the core CPEL model parameters ($\alpha$, $\beta$) for the CpG sites within that region.

```Python
import pandas as pd
import random
import numpy as np
import functions as F
# 1. Configure file paths
cpg_sites_file = "hg19_CpG.gz"
data = pd.read_csv('SRX8208802.mhap.gz',sep='\t',header=None,dtype={3:'str'},low_memory=False) 

# 2. Define parameters and genomic region 
chr='chr1'
start=10469
end = 11468
random.seed(42)
np.random.seed(42)

# 3. Run parameter estimation
n_list, theta, _ = F._cal_cpel(
    cpg_sites_file, data, chr=chr, start=start, end=end, step=250,
    compress=False, vis=False, filter_coverage=False
)
print((n_list, theta))

```
Output Result:
|  Sub-region | $R_1$ | $R_2$ | $R_3$ | $R_4$ |
| :--- | :---: | :---: | :---: | :---: |
| **No. of CpG sites** | 38 | 46 | 29 | 10 |
| **$\alpha$** | ` 0.586588` | `0.29507845` | `-4.99327237` | `3.74264579` |
| **$\beta$** | `0.28083722` *(Global)* | | | |

The table above is a historical output without coverage filtering. With filtering
enabled, a region that fails the thresholds returns an empty `theta` instead.

### Coverage filtering

`_cal_cpel` accepts these optional keyword arguments:

| Argument | Default | Meaning |
| --- | --- | --- |
| `filter_coverage` | `False` | Enable both coverage thresholds; off preserves existing calls. |
| `min_cov` | `5.0` | Minimum mean observations per CpG, inclusive. |
| `max_zero_frac` | `0.2` | Maximum fraction of CpGs without observations, inclusive. |

Mean coverage is the number of observed read-CpG entries divided by **all** CpGs
in the region, including uncovered CpGs. mHap read multiplicities are included.
Both thresholds must pass. Thresholds are evaluated before parameter estimation.
The return value remains `(n_list, theta, metrics)`. For a skipped region,
`theta=[]`, `metrics["skipped"]=True`, and `metrics["skip_reason"]` explains why.
Regions with no CpGs or no informative reads are skipped even when filtering is off.
Always check `metrics["skipped"]` before computing downstream model statistics.

Windows containing no CpG sites are omitted from `n_list`. A window containing
CpGs that have no reads is different: its CpGs remain in the coverage denominator.
The requested filter operates on the whole region, not separately on each window.

```Python
n_list, theta, metrics = F._cal_cpel(
    cpg_sites_file, data, chr=chr, start=start, end=end, step=250,
    compress=False, vis=False, filter_coverage=True,
    min_cov=5.0, max_zero_frac=0.2,
)
>>> if metrics["skipped"]:
...     print("Skipped:", metrics["skip_reason"])
print((n_list, theta, metrics))
```

Output Result:  
Skipped reasons: `mean_cov_below_min;zero_frac_above_max`

| Metric | Value | Description |
|---|---|---|
| `mean_cov` | 1.00813 | Average read coverage across all CpG sites in the region: the total number of observed CpG states across all reads within the region divided by the total number of CpG sites in the region. |
| `zero_cpg` | 66 | Number of CpG sites with zero read coverage. |
| `total_cpg` | 123 | Total number of CpG sites in the region, including those with zero read coverage. |
| `zero_frac` | 0.53659 | Proportion of CpG sites with zero read coverage, calculated as `zero_cpg / total_cpg`. |
| `skipped` | `True` | Parameter estimation was skipped because the region failed the coverage filters. |
| `skip_reason` | `mean_cov_below_min;zero_frac_above_max` | Average read coverage is below 5, and the proportion of CpG sites with zero read coverage exceeds 0.2. |





## Simulations
We provide a standalone module to run Monte Carlo simulations to validate the performance and statistical power of the newly introduced metrics in differential analysis.

```
Bash
# Usage: python Simulations.py <test_type: unmat/mat> <m_samples> <n_cpgs> <theta1> <theta2>
# Example1: Unpaired test, 5 samples per group, 4 CpGs, comparing theta1=[0.5,0.5] and theta2=[0.5,0.5]

python Simulations.py unmat 5 4 0.5,0.5 0.5,0.5

# Example2: Unpaired test, 5 samples per group, 4 CpGs, comparing theta1=[0.5,0.5] and theta2=[-0.5,0.5]

python Simulations.py unmat 5 4 0.5,0.5 -0.5,0.5

```
The resulting empirical cumulative distributions for different statistics are visualized below:
<img src="images\unmat_m5_n4_theta1_0.5_0.5_theta2_0.5_0.5_pvals.png" width="1024">
<img src="images\unmat_m5_n4_theta1_0.5_0.5_theta2_-0.5_0.5_pvals.png" width="1024">

## Relationship Between Statistics and Model Parameters
In this module, we generate heatmaps to visually demonstrate the relationship between the underlying CPEL model parameters ($\alpha$, $\beta$) and the various methylation statistics.By exploring this parameter grid ($\alpha, \beta \in [-8, 8]$), researchers can comprehensively understand how spatial concordance and methylation levels independently or synergistically drive different metrics (MML, NME, PDR, CHALM, and MCR).

### Reproducing the Analysis

```bash
# Step 1: Compute the metrics matrix over the parameter grid 
# (This caches the results into 'plots/computed_metrics.npz')
python Stat_para/Statistics_Parameters.py

# Step 2: Render and export the heatmaps (generates both individual and combined plots)
python Stat_para/plot.py
```
<p align="center">
  <img src="images/PDR.png" alt="PDR" width="30%">
  <img src="images/CHALM.png" alt="CHALM" width="30%">
  <img src="images/MCR.png" alt="MCR" width="30%">
</p>
<p align="center">
  <img src="images/MMl.png" alt="MML" width="30%">
  <img src="images/NME.png" alt="NME" width="30%">
</p>

## References
[1]  Henghui Li#, Jinglin Qi#, Rui Hou#, Leiqin Liu, Yaochen Xu, Xuejie Jin, Hongcang Gu*, Xiaoqi Zheng* and Jiantao Shi* Unified quantification of DNA methylation haplotype metrics under a discrete probability distribution model (submitted)
