"""Core coordinate conversion and ALE functions (NiMARE 0.5.0)."""

import numpy as np
from nilearn import image
from nimare.correct import FWECorrector
from nimare.io import convert_sleuth_to_dataset
from nimare.meta.cbma.ale import ALE
from nimare.utils import tal2mni
from scipy.stats import norm


def coordinates_to_mni(coordinates, space):
    """Convert an (N, 3) array once; leave already-MNI coordinates unchanged."""
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("Coordinates must have shape (N, 3).")
    space = space.strip().upper()
    if space in {"TAL", "TALAIRACH"}:
        return tal2mni(coordinates)
    if space == "MNI":
        return coordinates.copy()
    raise ValueError("Space must be MNI, TAL, or Talairach.")


def run_ale(sleuth_file, n_iters=10000, n_cores=1, seed=1234,
            cluster_forming_p=0.001, cluster_fwe_p=0.05):
    """Return the corrected result, thresholded raw Z map, and thresholded ALE map.

    Input: a run-ready MNI Sleuth TXT with one block per independent cohort.
    Merge eligible contrasts and repeated coordinates within each cohort before
    calling this function. Coordinate-free null studies are documented separately.
    """
    np.random.seed(seed)
    dataset = convert_sleuth_to_dataset(str(sleuth_file), target="ale_2mm")
    result = ALE().fit(dataset)
    corrected = FWECorrector(
        method="montecarlo", voxel_thresh=cluster_forming_p,
        n_iters=n_iters, n_cores=n_cores,
    ).transform(result)

    cluster_z = corrected.get_map(
        "z_desc-size_level-cluster_corr-FWE_method-montecarlo"
    )
    thresholded = image.threshold_img(
        cluster_z, threshold=norm.ppf(1 - cluster_fwe_p)
    )
    mask = image.math_img("img > 0", img=thresholded)
    z_map = image.math_img("mask * img", mask=mask, img=corrected.get_map("z"))
    ale_map = image.math_img("mask * img", mask=mask, img=corrected.get_map("stat"))
    return corrected, z_map, ale_map
