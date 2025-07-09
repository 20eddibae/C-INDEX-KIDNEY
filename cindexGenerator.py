import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import center_of_mass
from skimage import measure
import kits23


def load_nifti(path):
    """Load a NIfTI file and return its data, affine, and zooms."""
    img    = nib.load(path)
    data   = img.get_fdata().astype(int)
    affine = img.affine
    zooms  = img.header.get_zooms()
    return data, affine, zooms


def make_mask(seg, labels):
    """Create a binary mask from segmentation data for specified labels."""
    return np.isin(seg, labels).astype(np.uint8)


def get_center_of_mass(mask, affine):
    """Compute the center of mass of a binary mask in voxel coordinates."""
    voxel_com = center_of_mass(mask)
    # Convert to homogeneous coordinate and apply affine transform
    xyz_voxel = np.array([voxel_com[2], voxel_com[1], voxel_com[0], 1.0])
    xyz_world = affine.dot(xyz_voxel)
    return xyz_world[:3]


def compute_volume(mask, zooms):
    """Compute the volume of a binary mask in cubic centimeters."""
    vox_vol = np.prod(zooms)
    return np.sum(mask) * vox_vol / 1000.0


def sph_radius(volume_cc):
    """Compute the radius of a sphere with the given volume in centimeters."""
    return ((3 * volume_cc) / (4 * np.pi)) ** (1 / 3)


def c_index_for_masks(kid_mask, tum_mask, affine, zooms):
    """Compute the C-index for two binary masks."""
    kid_com = get_center_of_mass(kid_mask, affine)
    tum_com = get_center_of_mass(tum_mask, affine)

    kid_vol = compute_volume(kid_mask, zooms)
    tum_vol = compute_volume(tum_mask, zooms)

    kid_radius = sph_radius(kid_vol)
    tum_radius = sph_radius(tum_vol)

    distance = np.linalg.norm(kid_com - tum_com)
    c_index = 0.0 if distance == 0 else (kid_radius + tum_radius) / distance

    return {
        'com_kx': kid_com[0], 'com_ky': kid_com[1], 'com_kz': kid_com[2],
        'com_tx': tum_com[0], 'com_ty': tum_com[1], 'com_tz': tum_com[2],
        'volume_cc': tum_vol, 'radius_mm': tum_radius * 10,
        'distance_mm': distance * 10, 'c_index': c_index
    }


def reslice_to_coronal(vol):
    """Reslice a 3D volume to coronal view."""
    # Original vol shape: (Z, Y, X); coronal slices along Y-axis
    return vol.transpose(1, 0, 2)


def find_mid_polar_slice(kid_cor):
    """Find the mid-polar slice between ventral and dorsal kidney borders."""
    sums = kid_cor.sum(axis=(1, 2))
    indices = np.where(sums > 0)[0]
    if len(indices) == 0:
        return None
    return int((indices[0] + indices[-1]) // 2)


def centroid_2d(mask2d):
    """Compute the 2D centroid of a binary mask."""
    return center_of_mass(mask2d)


def find_largest_tumor_slice(tum_cor):
    """Find the slice index with the largest tumor cross-section."""
    sums = tum_cor.sum(axis=(1, 2))
    if np.all(sums == 0):
        return None
    return int(np.argmax(sums))


def compute_c_angle(kid_mask, tum_mask, affine, zooms):
    """Compute the C-angle for two binary masks."""
    # Reslice to coronal
    kid_cor = reslice_to_coronal(kid_mask)
    tum_cor = reslice_to_coronal(tum_mask)

    # Mid-polar slice
    mid_idx = find_mid_polar_slice(kid_cor)
    if mid_idx is None:
        return {'slice_mid_polar': None, 'slice_tumor': None,
                'angle_left_deg': None, 'angle_right_deg': None, 'c_angle_deg': None}

    # Reference point
    kid_slice = kid_cor[mid_idx]
    r0, c0 = centroid_2d(kid_slice)

    # Tumor slice
    tum_idx = find_largest_tumor_slice(tum_cor)
    if tum_idx is None:
        return {'slice_mid_polar': mid_idx, 'slice_tumor': None,
                'angle_left_deg': None, 'angle_right_deg': None, 'c_angle_deg': None}

    tum_slice = tum_cor[tum_idx]
    kid_slice_tum = kid_cor[tum_idx]
    tumor_in_kid = (tum_slice & kid_slice_tum).astype(np.uint8)

    # Extract contours
    contours = measure.find_contours(tumor_in_kid, 0.5)
    if not contours:
        return {'slice_mid_polar': mid_idx, 'slice_tumor': tum_idx,
                'angle_left_deg': None, 'angle_right_deg': None, 'c_angle_deg': None}

    coords = np.vstack(contours)
    rows, cols = coords[:, 0], coords[:, 1]
    delta_r = rows - r0
    delta_c = cols - c0

    # Compute angles to vertical axis
    angles = np.degrees(np.arctan2(np.abs(delta_c), np.abs(delta_r)))
    right_angles = angles[delta_c > 0]
    left_angles = angles[delta_c < 0]
    angle_right = float(np.max(right_angles)) if right_angles.size else 0.0
    angle_left = float(np.max(left_angles)) if left_angles.size else 0.0
    c_angle = angle_right + angle_left

    return {
        'slice_mid_polar': mid_idx,
        'slice_tumor': tum_idx,
        'angle_left_deg': angle_left,
        'angle_right_deg': angle_right,
        'c_angle_deg': c_angle
    }


def process_all_cases(data_dir):
    rows = []
    # data_dir should point at the KiTS23 “dataset” folder
    for case_name in os.listdir(data_dir):
        case_dir = os.path.join(data_dir, case_name)
        # skip anything that isn't a directory
        if not os.path.isdir(case_dir):
            continue

        seg_path = os.path.join(case_dir, "segmentation.nii.gz")
        # skip if this case has no segmentation
        if not os.path.exists(seg_path):
            continue

        # load the one file in each case folder
        seg, affine, zooms = load_nifti(seg_path)
        kid_mask = make_mask(seg, [1, 2, 3])
        tum_mask = make_mask(seg, [2])

        # compute  two metrics
        stats  = c_index_for_masks(kid_mask, tum_mask, affine, zooms)
        angles = compute_c_angle(kid_mask, tum_mask, affine, zooms)
        stats.update(angles)

        # tag with the case ID (folder name)
        stats['case_id'] = case_name
        rows.append(stats)

    return pd.DataFrame(rows)


if __name__ == '__main__':
    data_dir = '/Users/eddiebae/CS/C-INDEX-KIDNEY/KiTS23/dataset'
    df = process_all_cases(data_dir)
    df.to_csv('c_index_c_angle_results.csv', index=False)