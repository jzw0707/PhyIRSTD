"""Binary region overlap used for memory-reliability training labels."""

import numpy as np


def eval_i_u(annotation, segmentation, void_pixels=None):
    """Compute region similarity as the Jaccard Index.
    Arguments:
        annotation   (ndarray): binary annotation   map.
        segmentation (ndarray): binary segmentation map.
        void_pixels  (ndarray): optional mask with void pixels
    """
    assert annotation.shape == segmentation.shape, (
        f"Annotation({annotation.shape}) and segmentation:{segmentation.shape} dimensions do not match."
    )
    annotation = annotation.astype(bool)
    segmentation = segmentation.astype(bool)

    void_pixels = np.zeros_like(segmentation)

    # Intersection between all sets
    inters = np.sum(
        (segmentation & annotation) & np.logical_not(void_pixels), axis=(-2, -1)
    )
    union = np.sum(
        (segmentation | annotation) & np.logical_not(void_pixels), axis=(-2, -1)
    )
    return inters, union
