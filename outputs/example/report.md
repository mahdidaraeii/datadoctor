# Report: breast_cancer

569 rows, 31 columns. Settings: seed 42, row threshold 100000, column threshold 100.

## Executive summary

4 findings (1 medium, 2 low, 1 info).

## Figures

- **correlation_heatmap**: `outputs\correlation_heatmap.png`
- **target_distribution**: `outputs\target_distribution.png`
- **univariate_categorical_1**: `outputs\univariate_categorical_1.png`
- **univariate_numeric_1**: `outputs\univariate_numeric_1.png`
- **univariate_numeric_2**: `outputs\univariate_numeric_2.png`
- **univariate_numeric_3**: `outputs\univariate_numeric_3.png`

## Findings

### MEDIUM -- Extreme values in numeric columns (confidence 0.7)

- **Columns**: mean radius, mean texture, mean area, mean smoothness, mean compactness, mean symmetry, mean fractal dimension, radius error, texture error, perimeter error, area error, smoothness error, compactness error, concavity error, concave points error, symmetry error, fractal dimension error, worst perimeter, worst area, worst compactness, worst concavity, worst symmetry, worst fractal dimension
- **Evidence**: 23 columns have values more than 3 interquartile ranges beyond the quartiles and more than 3 standard deviations from the mean: fractal dimension error (10 values; quartiles 0.002248 and 0.004558, wide fences -0.004682 to 0.011488); symmetry error (9 values; quartiles 0.01516 and 0.02348, wide fences -0.0098 to 0.04844); perimeter error (8 values; quartiles 1.606 and 3.357, wide fences -3.647 to 8.61); radius error (7 values; quartiles 0.2324 and 0.4789, wide fences -0.5071 to 1.2184); smoothness error (7 values; quartiles 0.005169 and 0.008146, wide fences -0.003762 to 0.017077) and 18 more.
- **Interpretation**: Values this far from the rest are candidates for data errors, such as a wrong unit, a typo or a faulty sensor, or for real but rare events. They are candidates, not errors: one column alone cannot tell which.
- **Limitations**: The fences assume little about the shape of the data. The z-score assumes it is roughly bell-shaped and is pulled by the outliers it looks for. In skewed data both flag the long tail of a legitimate distribution. The cutoffs (1.5 and 3 interquartile ranges, |z| of 3) are conventions. Each column is examined alone, so a value that is unusual only in combination with others is not found. Values are not shown here; the lowest and highest are in the metrics.
- **Recommendation**: Look at the rows behind these values and decide whether each is an error or a real case. Do not remove them automatically.

### LOW -- Outlier candidates in numeric columns (confidence 0.4)

- **Columns**: mean perimeter, mean concavity, mean concave points, worst radius, worst texture, worst smoothness
- **Evidence**: 6 columns have values outside 1.5 interquartile ranges from the quartiles or more than 3 standard deviations from the mean: mean concavity (18 values, 3.2%: 18 by the IQR rule, 9 by z-score, 9 by both; quartiles 0.02956 and 0.1307, fences -0.12215 to 0.28241); worst radius (17 values, 3.0%: 17 by the IQR rule, 6 by z-score, 6 by both; quartiles 13.01 and 18.79, fences 4.34 to 27.46); mean perimeter (13 values, 2.3%: 13 by the IQR rule, 7 by z-score, 7 by both; quartiles 75.17 and 104.1, fences 31.775 to 147.49); mean concave points (10 values, 1.8%: 10 by the IQR rule, 6 by z-score, 6 by both; quartiles 0.02031 and 0.074, fences -0.060225 to 0.15453); worst smoothness (7 values, 1.2%: 7 by the IQR rule, 3 by z-score, 3 by both; quartiles 0.1166 and 0.146, fences 0.0725 to 0.1901) and 1 more.
- **Interpretation**: These values are unusual for their column, but mild candidates are common in real data. When many values are flagged, the column is usually skewed or heavy-tailed, which is a property of the distribution and not a fault. They are candidates, not errors.
- **Limitations**: The fences assume little about the shape of the data. The z-score assumes it is roughly bell-shaped and is pulled by the outliers it looks for. In skewed data both flag the long tail of a legitimate distribution. The cutoffs (1.5 and 3 interquartile ranges, |z| of 3) are conventions. Each column is examined alone, so a value that is unusual only in combination with others is not found. Values are not shown here; the lowest and highest are in the metrics.
- **Recommendation**: Check the columns with the highest rates first. Consider a transformation, such as a logarithm, for skewed columns before treating the tail as errors.

### LOW -- Numeric columns are strongly correlated (confidence 1.0)

- **Columns**: area error, compactness error, concavity error, fractal dimension error, mean area, mean compactness, mean concave points, mean concavity, mean perimeter, mean radius, mean smoothness, mean texture, perimeter error, radius error, worst area, worst compactness, worst concave points, worst concavity, worst fractal dimension, worst perimeter, worst radius, worst smoothness, worst texture
- **Evidence**: 44 pairs of columns have an absolute Pearson correlation of at least 0.8: mean radius / mean perimeter (r=1.00); worst radius / worst perimeter (r=0.99); mean radius / mean area (r=0.99); mean perimeter / mean area (r=0.99); worst radius / worst area (r=0.98) and 39 more.
- **Interpretation**: Two strongly correlated columns carry much of the same information. Keeping both in a model rarely helps and can make coefficients unstable or hard to interpret.
- **Limitations**: The 0.8 cutoff is a convention. Pearson correlation only measures a straight-line relationship, so a strong non-linear relationship can be missed.
- **Recommendation**: Consider dropping or combining one column from each pair, or using a model that handles correlated features well.

### INFO -- Columns most associated with the target (confidence 1.0)

- **Columns**: worst concave points, worst perimeter, mean concave points, worst radius, mean perimeter, worst area, mean radius, mean area, mean concavity, worst concavity, mean compactness, worst compactness, radius error, perimeter error, area error, worst texture, worst smoothness, worst symmetry, mean texture, concave points error, mean smoothness, mean symmetry, worst fractal dimension, compactness error, concavity error
- **Evidence**: 25 of the tested columns are associated with the target beyond a small effect: worst concave points (correlation ratio=0.79); worst perimeter (correlation ratio=0.78); mean concave points (correlation ratio=0.78); worst radius (correlation ratio=0.78); mean perimeter (correlation ratio=0.74) and 20 more.
- **Interpretation**: A numeric feature and the target are compared by Pearson correlation. A categorical feature or target is compared by the correlation ratio (a numeric feature grouped by a categorical one) or Cramer's V (two categorical columns), both on the same 0 to 1 scale as an absolute correlation. This is association, not causation, and says nothing about how a feature would behave in a model alongside the others.
- **Limitations**: Only pairs beyond a small effect size and significant after correcting for the number of features tested are listed. A column left out was not shown to be unrelated to the target, only not detected here.

## Limitations

This report combines schema profiling, data quality checks, exploratory analysis and ML-readiness diagnostics. It does not train or evaluate a model, and these checks do not cover every way a dataset can fail. Each finding's own Limitations field states what that specific check did not test; this section is about the report as a whole.
