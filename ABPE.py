import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from sklearn.decomposition import PCA
import nibabel as nib
import os


def build_smri_template_schemeC(label_vol,
                                spacing=(1.0, 1.0, 1.0),
                                alpha=0.45, beta=0.45, gamma=0.10,
                                eps=1e-6):

    label = np.array(label_vol, dtype=np.int32)
    sx, sy, sz = spacing
    X, Y, Z = label.shape

    xs = (np.arange(X) - 0.5) * sx
    ys = (np.arange(Y) - 0.5) * sy
    zs = (np.arange(Z) - 0.5) * sz
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing='ij')
    coords = np.stack([xx, yy, zz], axis=-1)  # [X,Y,Z,3]

    out = np.zeros((X, Y, Z), dtype=np.float32)

    labels = np.unique(label)
    labels = labels[labels > 0] 

    for rid in labels:
        mask = (label == rid)
        if mask.sum() == 0:
            continue

        pts = coords[mask]

        # ---------- centrality ----------
        centroid = pts.mean(axis=0)
        dists = np.linalg.norm(pts - centroid[None, :], axis=1)
        dmin, dmax = dists.min(), dists.max()
        if dmax - dmin < 1e-9:
            centrality_vals = np.ones_like(dists)
        else:
            dnorm = (dists - dmin) / (dmax - dmin + eps)
            centrality_vals = 1.0 - dnorm  

        # ---------- boundary distance ----------
        region_mask_uint8 = mask.astype(np.uint8)
        inside_dist = distance_transform_edt(region_mask_uint8, sampling=spacing)
        boundary_vals = inside_dist[mask]
        bmin, bmax = boundary_vals.min(), boundary_vals.max()
        if bmax - bmin < 1e-9:
            boundary_norm = np.ones_like(boundary_vals)
        else:
            boundary_norm = (boundary_vals - bmin) / (bmax - bmin + eps)

        # ---------- anisotropy ----------
        anisotropy_val = 0.0
        if pts.shape[0] >= 3:
            try:
                pca = PCA(n_components=3)
                pca.fit(pts)
                vars_ = pca.explained_variance_
                lam_sorted = np.sort(vars_)[::-1]
                lam1, lam3 = lam_sorted[0], lam_sorted[-1]
                anisotropy_val = float((lam1 - lam3) / (lam1 + eps))
            except Exception:
                anisotropy_val = 0.0
        anisotropy_arr = np.full_like(centrality_vals, anisotropy_val, dtype=np.float32)

        # ---------- combine ----------
        combined = alpha * centrality_vals + beta * boundary_norm + gamma * anisotropy_arr
        combined = np.clip(combined, 0.0, 1.0)

        out[mask] = combined

    return out


def process_and_save(input_nii, out_npy, out_nii):
    img = nib.load(input_nii)
    label_vol = img.get_fdata().astype(np.int32)
    spacing = img.header.get_zooms()[:3]

    template = build_smri_template_schemeC(label_vol, spacing=spacing)

    np.save(out_npy, template)

    out_img = nib.Nifti1Image(template.astype(np.float32), affine=img.affine)
    nib.save(out_img, out_nii)

    print(f" ✅ :\n  {out_npy}\n  {out_nii}")


# ---------------- 示例 ----------------
if __name__ == "__main__":
    input_file = r"path" 
    out_npy = r"path"
    out_nii = r"path"  

    process_and_save(input_file, out_npy, out_nii)
