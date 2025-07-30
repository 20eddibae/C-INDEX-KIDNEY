import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import center_of_mass
from skimage import measure
from scipy.spatial.distance import pdist


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


def compute_tumor_radius_mm(tum_mask, affine):
    """Compute tumor radius as max distance between any two tumor voxels divided by 2."""
    # Get coordinates of all tumor voxels
    tumor_coords = np.array(np.where(tum_mask > 0)).T
    
    if len(tumor_coords) < 2:
        return 0.0
    
    # For large tumors, sample points to avoid memory issues
    max_points = 1000
    if len(tumor_coords) > max_points:
        # Randomly sample points
        indices = np.random.choice(len(tumor_coords), max_points, replace=False)
        tumor_coords = tumor_coords[indices]
    
    # Convert to world coordinates (mm)
    from nibabel.affines import apply_affine
    tumor_coords_mm = apply_affine(affine, tumor_coords)
    
    # Calculate pairwise distances more efficiently
    # For large datasets, use a more memory-efficient approach
    if len(tumor_coords_mm) > 500:
        # Use a subset of points for radius calculation
        # Find the convex hull points which are likely to be the extremes
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(tumor_coords_mm)
            hull_points = tumor_coords_mm[hull.vertices]
            
            # Calculate distances between hull points
            max_distance = 0
            for i in range(len(hull_points)):
                for j in range(i+1, len(hull_points)):
                    dist = np.linalg.norm(hull_points[i] - hull_points[j])
                    if dist > max_distance:
                        max_distance = dist
        except:
            # Fallback: use a smaller random sample
            sample_size = min(100, len(tumor_coords_mm))
            indices = np.random.choice(len(tumor_coords_mm), sample_size, replace=False)
            sample_coords = tumor_coords_mm[indices]
            
            max_distance = 0
            for i in range(len(sample_coords)):
                for j in range(i+1, len(sample_coords)):
                    dist = np.linalg.norm(sample_coords[i] - sample_coords[j])
                    if dist > max_distance:
                        max_distance = dist
    else:
        # For smaller tumors, use the original approach
        from scipy.spatial.distance import pdist
        distances = pdist(tumor_coords_mm)
        max_distance = np.max(distances)
    
    # Get maximum distance and divide by 2 for radius
    radius = max_distance / 2.0
    
    return float(radius)


def compute_affected_kidney_mask(kid_mask, tum_mask):
    """Create affected kidney region: kidney voxels UNION with tumor voxels."""
    return (kid_mask | tum_mask).astype(np.uint8)


def compute_c_index_standard(kid_mask, tum_mask, affine, zooms):
    """Compute the STANDARD C-index: distance / tumor_radius (KiTS19 method)."""
    # Create affected kidney region (kidney UNION tumor)
    affected_kid_mask = compute_affected_kidney_mask(kid_mask, tum_mask)
    
    # Centers in mm
    affected_kid_com_mm = get_center_of_mass(affected_kid_mask, affine)
    tum_com_mm = get_center_of_mass(tum_mask, affine)

    # Tumor radius in mm (STANDARD METHOD)
    tumor_radius_mm = compute_tumor_radius_mm(tum_mask, affine)

    # Distance in mm
    distance_mm = np.linalg.norm(affected_kid_com_mm - tum_com_mm)

    c_index = distance_mm / tumor_radius_mm if tumor_radius_mm > 0 else np.nan

    return {
        'com_affected_kidney_x_mm': float(affected_kid_com_mm[0]),
        'com_affected_kidney_y_mm': float(affected_kid_com_mm[1]),
        'com_affected_kidney_z_mm': float(affected_kid_com_mm[2]),
        'com_tumor_x_mm': float(tum_com_mm[0]),
        'com_tumor_y_mm': float(tum_com_mm[1]),
        'com_tumor_z_mm': float(tum_com_mm[2]),
        'tumor_radius_mm': float(tumor_radius_mm),
        'distance_mm': float(distance_mm),
        'c_index_standard': float(c_index)
    }


def compute_c_index_kidney_radius(kid_mask, tum_mask, affine, zooms):
    """Compute the kidney-radius C-index: distance / kidney_radius (your original method)."""
    # Centers in mm
    kid_com_mm = get_center_of_mass(kid_mask, affine)
    tum_com_mm = get_center_of_mass(tum_mask, affine)

    # Kidney volume and radius
    kidney_volume_mm3 = np.sum(kid_mask) * np.prod(zooms)
    kidney_radius_mm = ((3 * kidney_volume_mm3) / (4 * np.pi)) ** (1/3)

    # Distance in mm
    distance_mm = np.linalg.norm(kid_com_mm - tum_com_mm)

    c_index = distance_mm / kidney_radius_mm if kidney_radius_mm > 0 else np.nan

    return {
        'kidney_volume_mm3': float(kidney_volume_mm3),
        'kidney_radius_mm': float(kidney_radius_mm),
        'c_index_kidney_radius': float(c_index)
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
        # kidney labels: 1=parenchyma, 3=cyst; tumor label: 2
        kid_mask = make_mask(seg, [1, 3])
        tum_mask = make_mask(seg, [2])

        # Only process if both kidney and tumor are present
        if np.sum(kid_mask) == 0 or np.sum(tum_mask) == 0:
            print(f"Warning: Skipping {case_id} - missing kidney or tumor")
            continue

        # Compute both C-index methods
        stats_standard = compute_c_index_standard(kid_mask, tum_mask, affine, zooms)
        stats_kidney = compute_c_index_kidney_radius(kid_mask, tum_mask, affine, zooms)
        angles = compute_c_angle(kid_mask, tum_mask, affine, zooms)
        
        # Combine all results
        stats = {**stats_standard, **stats_kidney, **angles}
        stats['case_id'] = case_id
        rows.append(stats)

    return pd.DataFrame(rows)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compute STANDARD C-index (tumor radius) and C-angle for KiTS23 cases')
    parser.add_argument('data_dir', help='Path to KiTS23 dataset folder')
    parser.add_argument('--out_csv', default='c_index_standard_results.csv',
                        help='Output CSV file')
    args = parser.parse_args()

    df = process_all_cases(args.data_dir)
    df.to_csv(args.out_csv, index=False)
    print(f"Results saved to {args.out_csv}")
    print(f"Standard C-index (tumor radius) computed for {len(df)} cases") 