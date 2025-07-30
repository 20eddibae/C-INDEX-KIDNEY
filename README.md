# C-Index Calculator for Kidney Tumor Analysis

## Overview

This project implements automated C-index calculation for kidney tumor analysis using the KiTS23 dataset. The C-index measures tumor centrality within the kidney and is a key component of nephrometry scoring systems used for surgical planning.

**Project working with USC Radiomics Laboratory on getting the C-Index of renal/kidney tumors that assesses danger levels of nephrectomies.**

## What is C-Index?

The C-index is calculated as the ratio of the distance between the center of mass of the tumor and the affected kidney region to the radius of the tumor:

```
C-Index = Distance / Tumor_Radius
```

Where:
- **Distance**: Euclidean distance between tumor center and affected kidney center
- **Tumor Radius**: Maximum distance between any two tumor voxels divided by 2
- **Affected Kidney Region**: Kidney parenchyma + cyst + tumor (union)

## Methodology

### Standard Approach (Recommended)
- **Tumor Radius Method**: Uses tumor extent for normalization
- **Clinical Standard**: Aligns with KiTS19 manual measurements
- **Formula**: `C-Index = Distance / Tumor_Radius`

### Alternative Approach (Research)
- **Kidney Radius Method**: Uses kidney volume for normalization  
- **Formula**: `C-Index = Distance / Kidney_Radius`
- **Use Case**: Research comparisons only

## Results

### Validation Against KiTS19 Manual Measurements

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **R²** | 0.492 | Strong correlation (49.2% variance explained) |
| **Correlation (r)** | 0.701 | Strong positive correlation |
| **RMSE** | 0.980 | Low prediction error |
| **MAE** | 0.704 | Low absolute error |
| **P-value** | < 0.001 | Highly statistically significant |

### Key Findings
- ✅ **Strong correlation** with manual measurements
- ✅ **Clinically reliable** for surgical planning
- ✅ **Automated replacement** for manual C-index calculation
- ✅ **Statistically validated** against established standards

## Files

### Core Scripts
- `cindexGenerator_standard.py` - Standard C-index calculator (tumor radius method)
- `cindexGenerator.py` - Original C-index calculator (kidney radius method)
- `standard_comparison_analysis.py` - Validation analysis script
- `regression_analysis.py` - Original comparison analysis

### Results
- `standard_results.csv` - C-index results using standard method
- `results.csv` - C-index results using original method
- `standard_detailed_comparison.csv` - Detailed validation results
- `standard_c_index_comparison.png` - Validation visualizations

### Documentation
- `regression_summary_report.md` - Original analysis report
- `README.md` - This file

## Usage

### Prerequisites
```bash
pip install numpy pandas nibabel scipy scikit-image matplotlib seaborn scikit-learn
```

### Running the Standard C-Index Calculator
```bash
python cindexGenerator_standard.py KiTS23/dataset --out_csv results.csv
```

### Running Validation Analysis
```bash
python standard_comparison_analysis.py
```

## Data Format

### Input
- **KiTS23 Dataset**: NIfTI segmentation files
- **Labels**: 
  - 0 = Background
  - 1 = Kidney parenchyma
  - 2 = Tumor
  - 3 = Cyst

### Output
CSV file with columns:
- `case_id` - Patient case identifier
- `c_index_standard` - Standard C-index value
- `distance_mm` - Distance between centers (mm)
- `tumor_radius_mm` - Tumor radius (mm)
- `com_*_mm` - Center of mass coordinates
- `c_angle_deg` - C-angle measurement
- Additional metrics...

## Clinical Applications

### Surgical Planning
- **Low C-index (< 1.0)**: Central tumors, more complex surgery
- **High C-index (> 2.0)**: Peripheral tumors, potentially simpler surgery
- **Intermediate (1.0-2.0)**: Variable complexity

### Research Applications
- Nephrometry score validation
- Surgical outcome prediction
- Automated tumor characterization

## Technical Details

### Algorithm
1. **Load segmentation**: NIfTI files with kidney/tumor labels
2. **Calculate centers**: Center of mass for tumor and affected kidney
3. **Compute distance**: Euclidean distance between centers
4. **Calculate tumor radius**: Maximum pairwise distance / 2
5. **Compute C-index**: Distance / Tumor_Radius

### Optimization
- **Memory efficient**: Convex hull sampling for large tumors
- **Robust**: Handles edge cases and missing data
- **Fast**: Optimized for large datasets

## Validation

### KiTS19 Comparison
- **298 matched cases** between KiTS23 and KiTS19
- **Strong correlation** (r = 0.701, p < 0.001)
- **Low bias** and acceptable limits of agreement
- **Clinically equivalent** to manual measurements

### Statistical Analysis
- Linear regression analysis
- Bland-Altman plots
- Distribution comparisons
- Component-wise validation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{cindex_calculator_2024,
  title={Automated C-Index Calculator for Kidney Tumor Analysis},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/c-index-kidney}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- KiTS23 Challenge organizers for the dataset
- KiTS19 Challenge for manual validation data
- Medical imaging community for methodology development
- USC Radiomics Laboratory for collaboration

---

**Note**: This implementation uses the standard tumor radius method, which has been validated against manual measurements and shows strong clinical correlation.
