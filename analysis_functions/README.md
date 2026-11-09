# Core analysis functions

This folder contains compact functions for the RDoC social-processing
meta-analysis in schizophrenia: ALE, Neurosynth ROI decoding, and MACM.
The functions were extracted and simplified from the analysis scripts. They
preserve the core analytical settings but are not the full execution pipeline.
No analysis runs automatically when a module is imported.

Study screening, contrast selection, input tables, datasets, result maps, HPC
submission scripts, logging, and manuscript preparation are not included.
FSN, GCLDA, and subtraction analyses are also outside this folder's scope.
Reproducing the reported findings requires the same curated inputs, database,
and software versions, in addition to these functions.

## Files and environments

| File | Main functions |
| --- | --- |
| `ale.py` | Coordinate conversion; ALE; cluster-size FWE correction |
| `decoding.py` | Cluster masks; Neurosynth decoding; term ranking; word clouds |
| `macm.py` | Seed extraction; record selection; MKDAChi2; map filtering; intersections |

Use Python 3.10 and **separate environments**:

```bash
# In the ALE/decoding environment:
python -m pip install -r requirements_ale_decoding.txt

# In a different MACM environment:
python -m pip install -r requirements_macm.txt
```

ALE and decoding use NiMARE 0.5.0. MACM uses NiMARE 0.22.0. The dependency
lists record key versions; they are not complete lockfiles. Do not install both
lists into the same environment. The original MACM jobs limited BLAS libraries
to one thread per worker; CPU allocation and job scheduling are left to the caller.

## 1. ALE

Prepare a run-ready Sleuth TXT with MNI coordinates and one block per independent
cohort. Pool eligible contrasts from the same cohort within that block and remove
identical repeated coordinates within the block. The sample-size field uses the
curated smaller group size (MIN_N), not the sum of patient and control groups.
These input-preparation rules are not performed by `run_ale`.

Coordinate-free null studies belong in the study documentation, not in the
run-ready TXT parsed by this function. Analyses in the study required at least
17 coordinate-bearing independent blocks. The same ALE function can be applied
to separately prepared overall, subdomain, direction-specific, single-annotation,
and sensitivity inputs.

`coordinates_to_mni` accepts `MNI`, `TAL`, or `Talairach` (case-insensitive).
Talairach coordinates use NiMARE's Lancaster-based `tal2mni`; convert only once,
then label the resulting TXT as MNI.

Defaults: 10,000 permutations, cluster-forming p = 0.001, cluster-size FWE
p = 0.05, random seed 1234. The thresholded Z output contains the original ALE
Z values within surviving clusters, not the cluster-corrected Z values.

```python
from ale import run_ale

result, z_map, ale_map = run_ale("ALL_hypo.txt", n_cores=4)
z_map.to_filename("ALL_hypo_z_thresh.nii.gz")
ale_map.to_filename("ALL_hypo_stat_thresh.nii.gz")
```

## 2. Neurosynth decoding

Inputs: the prepared Neurosynth Dataset, each significant ALE cluster mask,
and Cognitive Atlas concept names (the `name` column of the concept-name CSV).
The analysis database contained 14,371 records and 3,228 tested terms; 138 terms
matched the Cognitive Atlas list. Supply the same resources to reproduce the
rankings. No dataset or concept-name file is redistributed here.

`cluster_masks` separates the positive thresholded ALE map into complete
6-connected clusters, ordered by size. `decode_roi` resamples each mask to the
database grid with nearest-neighbour interpolation. Records with a coordinate
inside the ROI are compared with the remaining database records.

Settings: `terms_abstract_tfidf`, annotation threshold 0.001, prior 0.5.
The annotation threshold is not a statistical threshold for displaying words.
After exact Cognitive Atlas name matching and term deduplication, terms are
ranked by `probReverse`; the top 20 are selected without p/q/Z filtering.
The original script also exported reference-only q-values; this compact version
omits that unused calculation. It does not change term selection or weights.

```python
import pandas as pd
from nimare.dataset import Dataset
from decoding import cluster_masks, decode_roi, make_wordcloud

dataset = Dataset.load("neurosynth_dataset.pkl.gz")
names = pd.read_csv("cogat_aliases.csv")["name"].dropna()
roi = cluster_masks("SC_hypo_z_thresh.nii.gz")[0]
decoded = decode_roi(dataset, roi, names)
decoded["top_terms"].to_csv("SC_hypo_top20.csv", index=False)
cloud = make_wordcloud(decoded["top_terms"])
cloud.to_file("SC_hypo_top20.png")
```

For a brain-shaped cloud, pass a 2D mask (255 outside, 0 inside) and an optional
font path. These artwork assets and the original background overlay are not
included. Check `cloud.layout_` if a chosen shape/font cannot fit all 20 words.
Word size encodes database-based reverse-inference probability, not the probability
or severity of impairment in patients. Colours are decorative.

## 3. MACM and intersections

Input: the prepared MNI Neurosynth Dataset and its original whole-brain mask,
plus the thresholded ALE maps. Use each complete cluster as a seed, not a sphere
around its peak:

| Seed | Source map | Peak MNI (mm) |
| --- | --- | --- |
| SC right amygdala | SC_hypo | 28, -8, -16 |
| PUO right middle temporal/angular, cluster 1 | PUO_hypo | 52, -56, 14 |
| PUO mPFC, cluster 2 | PUO_hypo | 0, 56, 18 |

For each seed, compare seed-selected with remaining database records using
`MKDAChi2`, a 10-mm binary spherical kernel, and `prior=None`. This kernel models
activation around database coordinates; it does not enlarge the ALE seed.
The database is not restricted to social tasks or to a particular diagnosis.

Correction: 10,000 label permutations, whole-brain voxel-FWE p < 0.05, random
seed 1234. Positive associations mean more frequent reported activation in
seed-selected records. The corrector also calculates auxiliary cluster statistics,
but these do not determine the reported voxel-FWE maps.

```python
from nimare.dataset import Dataset
from macm import cluster_seed, run_macm, filter_positive_clusters

dataset = Dataset.load("neurosynth_dataset.pkl.gz")
seed = cluster_seed("SC_hypo_z_thresh.nii.gz", [28, -8, -16])
macm = run_macm(dataset, seed, analysis_mask="neurosynth_analysis_mask.nii.gz",
                n_cores=4)
positive = filter_positive_clusters(macm["positive"], min_voxels=100)
positive.to_filename("SC_hypo_C01_positive_vFWE05_filter100.nii.gz")
```

Repeat the seed-specific analysis for both PUO clusters. Remove clusters smaller
than 100 voxels using 26-neighbour connectivity from each positive map before
intersection. There is no additional Z > 3.29 cutoff. The three seed maps must
have the same shape and affine.

```python
from macm import conjunction

# Each entry is a positive voxel-FWE map already filtered at 100 voxels.
common = conjunction(["SC_filtered.nii.gz", "PUO1_filtered.nii.gz",
                      "PUO2_filtered.nii.gz"])
common.to_filename("three_seed_conjunction_minZ.nii.gz")
# Pass any two maps for a pairwise intersection.
```

The intersection retains the minimum Z at overlapping positive voxels. No further
extent filter is applied to the intersection. These maps describe shared
literature-based coactivation, not a new group-difference test or patient-specific
connectivity. Records shared across seeds remain eligible for each seed MACM.

The decoding and MACM functions intentionally preserve their original voxel-index
conventions: integer truncation in NiMARE 0.5.0 and nearest-voxel rounding in
NiMARE 0.22.0, respectively.

## Validation of the compact functions

The six ROI top-20 term lists and weights, three MACM seed selections, three
extent-filtered positive maps, and four intersections were checked against the
existing analysis outputs. ALE and MKDAChi2 also passed small synthetic
fit/correction tests. The full 10,000-permutation analyses were not rerun when
preparing this compact code folder.

## Implementation references

- [NiMARE ALE API (0.5.0)](https://nimare.readthedocs.io/en/0.5.0/generated/nimare.meta.cbma.ale.ALE.html)
- [NiMARE NeurosynthDecoder API (0.5.0)](https://nimare.readthedocs.io/en/0.5.0/generated/nimare.decode.discrete.NeurosynthDecoder.html)
- [NiMARE source](https://github.com/neurostuff/NiMARE)
- [Neurosynth data](https://github.com/neurosynth/neurosynth-data)
- [Cognitive Atlas](https://www.cognitiveatlas.org/)

These wrappers call the cited libraries; they do not reimplement ALE, Neurosynth
decoding, or MKDAChi2. Repository-level licensing and citation information should
be retained when this folder is added to an existing repository.
