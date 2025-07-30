import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import center_of_mass
from skimage import measure


def load_nifti(path):
    """Load a NIfTI file and return its data, affine, and voxel spacings."""
    img = nib.load(path)
    data = img.get_fdata().astype(np.uint8)
    affine = img.affine
    zooms = img.header.get_zooms()
    return data, affine, zooms


def make_mask(seg, labels):
    """Create a binary mask from segmentation data for specified labels."""
    return np.isin(seg, labels).astype(np.uint8)


def get_center_of_mass(mask, affine):
    """Compute the center of mass of a binary mask in world (mm) coordinates."""
    voxel_com = center_of_mass(mask)
    # Apply affine to convert voxel COM to world coordinates (mm)
    from nibabel.affines import apply_affine
    return apply_affine(affine, voxel_com)


def compute_volume_mm3(mask, zooms):
    """Compute the volume of a binary mask in cubic millimeters."""
    vox_vol_mm3 = np.prod(zooms)
    return np.sum(mask) * vox_vol_mm3


def sph_radius_mm(volume_mm3):
    """Compute the radius (mm) of a sphere with the given volume (mm^3)."""
    return ((3 * volume_mm3) / (4 * np.pi)) ** (1 / 3)


def compute_c_index(kid_mask, tum_mask, affine, zooms):
    """Compute the classical C-index: distance / kidney_radius."""
    # Centers in mm
    kid_com_mm = get_center_of_mass(kid_mask, affine)
    tum_com_mm = get_center_of_mass(tum_mask, affine)

    # Volumes in mm^3
    V_kid_mm3 = compute_volume_mm3(kid_mask, zooms)
    R_kid_mm = sph_radius_mm(V_kid_mm3)

    # Distance in mm
    distance_mm = np.linalg.norm(kid_com_mm - tum_com_mm)

    c_index = distance_mm / R_kid_mm if R_kid_mm > 0 else np.nan

    return {
        'com_kx_mm': float(kid_com_mm[0]),
        'com_ky_mm': float(kid_com_mm[1]),
        'com_kz_mm': float(kid_com_mm[2]),
        'com_tx_mm': float(tum_com_mm[0]),
        'com_ty_mm': float(tum_com_mm[1]),
        'com_tz_mm': float(tum_com_mm[2]),
        'kidney_volume_mm3': float(V_kid_mm3),
        'kidney_radius_mm': float(R_kid_mm),
        'distance_mm': float(distance_mm),
        'c_index': float(c_index)
    }


def reslice_to_coronal(vol):
    """Reslice a 3D volume to coronal view (Y, Z, X)."""
    return vol.transpose(1, 0, 2)


def find_mid_polar_slice(kid_cor):
    sums = kid_cor.sum(axis=(1, 2))
    idx = np.where(sums > 0)[0]
    return int((idx[0] + idx[-1]) // 2) if idx.size else None


def centroid_2d(mask2d):
    return center_of_mass(mask2d)


def find_largest_tumor_slice(tum_cor):
    sums = tum_cor.sum(axis=(1, 2))
    return int(np.argmax(sums)) if np.any(sums) else None


def compute_c_angle(kid_mask, tum_mask, affine, zooms):
    """Compute the C-angle for two binary masks."""
    # Reslice
    kid_cor = reslice_to_coronal(kid_mask)
    tum_cor = reslice_to_coronal(tum_mask)

    mid_idx = find_mid_polar_slice(kid_cor)
    if mid_idx is None:
        return dict(slice_mid_polar=None, slice_tumor=None,
                    angle_left_deg=None, angle_right_deg=None, c_angle_deg=None)

    # Reference centroid on mid-polar slice
    kid_slice = kid_cor[mid_idx]
    r0, c0 = centroid_2d(kid_slice)

    tum_idx = find_largest_tumor_slice(tum_cor)
    if tum_idx is None:
        return dict(slice_mid_polar=mid_idx, slice_tumor=None,
                    angle_left_deg=None, angle_right_deg=None, c_angle_deg=None)

    tumor_slice = tum_cor[tum_idx]
    kidney_slice = kid_cor[tum_idx]
    overlap = (tumor_slice & kidney_slice)

    contours = measure.find_contours(overlap, 0.5)
    if not contours:
        return dict(slice_mid_polar=mid_idx, slice_tumor=tum_idx,
                    angle_left_deg=None, angle_right_deg=None, c_angle_deg=None)

    coords = np.vstack(contours)
    dr = coords[:, 0] - r0
    dc = coords[:, 1] - c0
    angles = np.degrees(np.arctan2(np.abs(dc), np.abs(dr)))
    right_angles = angles[dc > 0]
    left_angles  = angles[dc < 0]

    angle_r = float(right_angles.max()) if right_angles.size else 0.0
    angle_l = float(left_angles.max())  if left_angles.size  else 0.0
    c_angle = angle_r + angle_l

    return {
        'slice_mid_polar': mid_idx,
        'slice_tumor': tum_idx,
        'angle_left_deg': angle_l,
        'angle_right_deg': angle_r,
        'c_angle_deg': c_angle
    }


def process_all_cases(data_dir):
    rows = []
    for case_id in sorted(os.listdir(data_dir)):
        case_dir = os.path.join(data_dir, case_id)
        if not os.path.isdir(case_dir):
            continue

        seg_path = os.path.join(case_dir, 'segmentation.nii.gz')
        if not os.path.exists(seg_path):
            continue

        seg, affine, zooms = load_nifti(seg_path)
        # According to plans: kidney_mask = (seg == 1) | (seg == 2) | (seg == 3)
        # This includes kidney + tumor + cyst
        kid_mask = make_mask(seg, [1, 2, 3])
        tum_mask = make_mask(seg, [2])

        stats = compute_c_index(kid_mask, tum_mask, affine, zooms)
        angles = compute_c_angle(kid_mask, tum_mask, affine, zooms)
        stats.update(angles)
        stats['case_id'] = case_id
        rows.append(stats)

    return pd.DataFrame(rows)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compute C-index and C-angle for KiTS23 cases')
    parser.add_argument('data_dir', help='Path to KiTS23 dataset folder')
    parser.add_argument('--out_csv', default='c_index_c_angle_results.csv',
                        help='Output CSV file')
    args = parser.parse_args()

    df = process_all_cases(args.data_dir)
    df.to_csv(args.out_csv, index=False)
    print(f"Results saved to {args.out_csv}")
