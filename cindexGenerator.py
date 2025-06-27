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
    com = center_of_mass(mask)
    result = np.dot(affine, mask)
    return result
def compute_volume(mask, zooms):
    """Compute the volume of a binary mask in cubic centimeters."""
    vox_vol = np.prod(zooms)
    return np.sum(mask) * vox_vol / 1000.0
def sph_radius(Volume): 
    """Compute the radius of a sphere with the given volume in centimeters."""
    return (3 * Volume * 1000 / (4 * np.pi)) ** (1 / 3)
def c_index_for_masks(kid_mask, tum_mask, affine, zooms):
    """Compute the C-index for two binary masks."""
    kid_com = get_center_of_mass(kid_mask, affine)
    tum_com = get_center_of_mass(tum_mask, affine)

    kid_vol = compute_volume(kid_mask, zooms)
    tum_vol = compute_volume(tum_mask, zooms)

    kid_radius = sph_radius(kid_vol)
    tum_radius = sph_radius(tum_vol)

    distance = np.linalg.norm(kid_com - tum_com)
    
    if distance == 0:
        return 0.0

    c_index = (kid_radius + tum_radius) / distance
    return {
      'com_kx': kid_com[0], 'com_ky': kid_com[1], 'com_kz': kid_com[2],
      'com_tx': tum_com[0], 'com_ty': tum_com[1], 'com_tz': tum_com[2],
      'volume_cc': tum_vol, 'radius_mm': tum_radius * 10,
      'distance_mm': distance, 'c_index': c_index
    }
 

def process_all_cases(data_dir):
    rows = []
    for fname in os.listdir(data_dir):
        if not fname.endswith('.nii.gz'): continue
        case_id = fname.replace('.nii.gz','')
        seg, affine, zooms = load_nifti(os.path.join(data_dir,fname))
        kid_mask = make_mask(seg, [1,2,3])
        tum_mask = make_mask(seg, [2])
        stats = c_index_for_masks(kid_mask, tum_mask, affine, zooms)
        stats['case_id'] = case_id
        rows.append(stats)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    df = process_all_cases(data_dir)
    df.to_csv('c_index_results.csv', index=False)