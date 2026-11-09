"""Seed-specific MKDAChi2 MACM and positive-map intersections (NiMARE 0.22.0)."""

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from nilearn.image import load_img, math_img
from nilearn.maskers import NiftiMasker
from nimare.correct import FWECorrector
from nimare.meta.cbma.mkda import MKDAChi2
from scipy.ndimage import generate_binary_structure, label


def cluster_seed(thresholded_z_map, peak_mni):
    """Use the entire 6-connected ALE cluster containing a specified MNI peak."""
    img = load_img(thresholded_z_map)
    labels, _ = label(img.get_fdata() > 0, generate_binary_structure(3, 1))
    ijk = np.rint(nib.affines.apply_affine(np.linalg.inv(img.affine), peak_mni)).astype(int)
    if np.any(ijk < 0) or np.any(ijk >= np.array(img.shape)):
        raise ValueError("Peak is outside the image grid.")
    cluster = labels[tuple(ijk)]
    if cluster == 0:
        raise ValueError("Peak is outside the significant clusters.")
    return nib.Nifti1Image((labels == cluster).astype("uint8"), img.affine)


def select_seed_records(dataset, seed_mask):
    """Select records with at least one reported coordinate inside the seed."""
    reference = dataset.masker.mask_img
    mask = resample_from_to(
        load_img(seed_mask), (reference.shape, reference.affine), order=0
    ).get_fdata() > 0
    coords = dataset.coordinates
    inverse = np.linalg.inv(reference.affine)
    xyz = coords[["x", "y", "z"]].to_numpy()
    voxels = np.einsum("ij,nj->ni", inverse[:3, :3], xyz) + inverse[:3, 3]
    ijk = np.rint(voxels).astype(int)  # NiMARE 0.22.0 nearest-voxel convention.
    valid = ((ijk >= 0) & (ijk < np.array(mask.shape))).all(axis=1)
    hits = np.zeros(len(coords), dtype=bool)
    hits[valid] = mask[tuple(ijk[valid].T)]
    return set(coords.loc[hits, "id"])


def run_macm(dataset, seed_mask, analysis_mask=None, n_iters=10000,
             n_cores=1, seed=1234, alpha=0.05):
    """Return corrected results, signed significant maps, and both record sets.

    Refit the dataset masker to avoid legacy serialized-masker incompatibilities.
    Use the original database mask/grid; do not substitute a seed mask here.
    """
    mask = analysis_mask if analysis_mask is not None else dataset.masker.mask_img
    dataset.masker = NiftiMasker(mask_img=load_img(mask)).fit()
    selected = select_seed_records(dataset, seed_mask)
    remaining = set(dataset.ids) - selected
    if not selected or not remaining:
        raise ValueError("Both selected and remaining database records are required.")
    estimator = MKDAChi2(
        kernel__r=10, prior=None, fwe_null_method="label-permutation",
        random_state=seed,
    )
    result = estimator.fit(dataset.slice(sorted(selected)), dataset.slice(sorted(remaining)))
    corrected = FWECorrector(
        method="montecarlo", n_iters=n_iters, n_cores=n_cores, voxel_thresh=0.001,
    ).transform(result)
    # voxel_thresh above only affects auxiliary cluster calculations, not this voxel-FWE test.
    suffix = "_level-voxel_corr-FWE_method-montecarlo"
    p = corrected.get_map("p_desc-association" + suffix).get_fdata()
    z_img = corrected.get_map("z_desc-association" + suffix)
    z = z_img.get_fdata()
    significant = np.isfinite(p) & np.isfinite(z) & (p < alpha)
    positive = nib.Nifti1Image(np.where(significant & (z > 0), z, 0), z_img.affine)
    negative = nib.Nifti1Image(np.where(significant & (z < 0), z, 0), z_img.affine)
    return {"result": corrected, "positive": positive, "negative": negative,
            "selected_ids": sorted(selected), "remaining_ids": sorted(remaining)}


def filter_positive_clusters(positive_map, min_voxels=100):
    """Remove small 26-connected clusters before intersection; no extra Z cutoff."""
    img = load_img(positive_map)
    data = img.get_fdata()
    labels, _ = label(data > 0, generate_binary_structure(3, 3))
    sizes = np.bincount(labels.ravel())
    keep = (labels > 0) & (sizes[labels] >= min_voxels)
    return nib.Nifti1Image(np.where(keep, data, 0), img.affine)


def conjunction(positive_maps):
    """Intersect two or more aligned positive maps and retain the minimum Z.

    Input maps must already be voxel-FWE thresholded and extent-filtered.
    This is a spatial intersection, not a new statistical group comparison.
    """
    images = [load_img(img) for img in positive_maps]
    if len(images) < 2:
        raise ValueError("At least two maps are required.")
    common = images[0]
    for img in images[1:]:
        common = math_img(
            "np.where((a > 0) & (b > 0), np.minimum(a, b), 0)", a=common, b=img
        )
    return common
