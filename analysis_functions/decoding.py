"""ROI-based Neurosynth decoding and top-20 term display (NiMARE 0.5.0)."""

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from nilearn.image import load_img
from nimare.decode.discrete import NeurosynthDecoder
from scipy.ndimage import generate_binary_structure, label


def cluster_masks(thresholded_z_map):
    """Split a positive ALE map into complete 6-connected clusters, largest first."""
    img = load_img(thresholded_z_map)
    labels, count = label(img.get_fdata() > 0, generate_binary_structure(3, 1))
    order = sorted(range(1, count + 1), key=lambda i: (-int((labels == i).sum()), i))
    return [nib.Nifti1Image((labels == i).astype("uint8"), img.affine) for i in order]


def decode_roi(dataset, roi_mask, cognitive_atlas_names, top_n=20):
    """Return all terms, matched/ranked terms, top terms, and selected record IDs.

    Match Cognitive Atlas names exactly. Rank by probReverse without p/q/Z
    filtering. probReverse is a database-based association, not patient impairment.
    """
    reference = dataset.masker.mask_img
    roi = resample_from_to(
        load_img(roi_mask), (reference.shape, reference.affine), order=0
    ).get_fdata() > 0
    coords = dataset.coordinates
    xyz = coords[["x", "y", "z"]].to_numpy()
    inverse = np.linalg.inv(reference.affine)
    # Preserve the NiMARE 0.5.0 truncation convention used in the original decoding.
    ijk = (np.einsum("ij,nj->ni", inverse[:3, :3], xyz) + inverse[:3, 3]).astype(int)
    valid = ((ijk >= 0) & (ijk < np.array(roi.shape))).all(axis=1)
    hits = np.zeros(len(coords), dtype=bool)
    hits[valid] = roi[tuple(ijk[valid].T)]
    selected = sorted(coords.loc[hits, "id"].unique())
    if not selected or set(selected) == set(dataset.ids):
        raise ValueError("Both selected and remaining database records are required.")

    decoder = NeurosynthDecoder(
        feature_group="terms_abstract_tfidf", frequency_threshold=0.001,
        prior=0.5, correction=None,
    )
    decoder.fit(dataset)
    terms = decoder.transform(ids=selected).rename_axis("feature").reset_index()
    terms.insert(1, "term", terms["feature"].str.split("__", n=1).str[-1])
    terms["cognitive_atlas_match"] = terms["term"].isin(set(cognitive_atlas_names))
    eligible = terms["cognitive_atlas_match"] & np.isfinite(terms["probReverse"])
    ranked = terms.loc[eligible].sort_values(
        ["probReverse", "term"], ascending=[False, True]
    ).drop_duplicates("term")
    return {"all_terms": terms, "ranked_terms": ranked,
            "top_terms": ranked.head(top_n), "selected_ids": selected}


def make_wordcloud(top_terms, mask=None, font_path=None):
    """Return a top-20 WordCloud; optional mask uses white (255) outside the shape."""
    from wordcloud import WordCloud

    chosen = top_terms.sort_values(
        ["probReverse", "term"], ascending=[False, True]
    ).drop_duplicates("term").head(20)
    weights = dict(zip(chosen["term"], chosen["probReverse"].astype(float)))
    return WordCloud(
        width=2000, height=2000, background_color=None, mode="RGBA",
        mask=mask, font_path=font_path, max_words=20, max_font_size=800,
        min_font_size=36, random_state=20, font_step=2, margin=0,
        colormap="viridis", repeat=False,
    ).generate_from_frequencies(weights)
