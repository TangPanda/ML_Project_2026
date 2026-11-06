from pathlib import Path
import numpy as np
import tifffile as tiff

LABEL_FILE = Path(
    r"C:\Users\abreun01\OneDrive - NYU Langone Health\MSCI\FALL 2026\Machine Learning\Project\LIVECell\knn_segmentation_output\A172_Phase_C7_1_00d00h00m_1_instance_labels.tif"
)

labels = tiff.imread(LABEL_FILE)

print("File:", LABEL_FILE.name)
print("Shape:", labels.shape)
print("Data type:", labels.dtype)
print("Minimum label:", labels.min())
print("Maximum label / predicted cell count:", labels.max())
print("Nonzero (cell) pixels:", np.count_nonzero(labels))
print("Total pixels:", labels.size)
print("Percent foreground:", 100 * np.count_nonzero(labels) / labels.size)

unique_labels = np.unique(labels)

print("\nNumber of unique values:", len(unique_labels))
print("First 25 label values:", unique_labels[:25])
print("Last 25 label values:", unique_labels[-25:])

# Confirm whether labels are sequential: 0, 1, 2, ..., N
expected_labels = np.arange(labels.max() + 1)
print("\nLabels are sequential from 0 to max:",
      np.array_equal(unique_labels, expected_labels))