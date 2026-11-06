from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# FILE LOCATIONS
# ============================================================

OUTPUT_DIR = Path(
    r"C:\Users\abreun01\OneDrive - NYU Langone Health\MSCI\FALL 2026\Machine Learning\Project\LIVECell\knn_test_segmentation_output"
)

RESULTS_CSV = OUTPUT_DIR / "test_evaluation_results.csv"

PLOT_FILE = OUTPUT_DIR / "count_error_boxplot_by_tissue_and_cell_type.png"

SUMMARY_FILE = OUTPUT_DIR / "count_error_summary_by_cell_type.csv"


# ============================================================
# LOAD AND VALIDATE RESULTS
# ============================================================

if not RESULTS_CSV.exists():
    raise FileNotFoundError(
        f"Could not find the results file:\n{RESULTS_CSV}\n\n"
        "Update OUTPUT_DIR so it matches the folder containing "
        "test_evaluation_results.csv."
    )

df = pd.read_csv(RESULTS_CSV)

required_columns = {"cell_line", "count_error"}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required column(s): {sorted(missing_columns)}\n\n"
        f"Columns available in the CSV:\n{list(df.columns)}"
    )

# Retain only cell-line names and valid numeric count errors.
plot_df = df[["cell_line", "count_error"]].copy()

plot_df["count_error"] = pd.to_numeric(
    plot_df["count_error"],
    errors="coerce"
)

plot_df = plot_df.dropna(subset=["cell_line", "count_error"])

if plot_df.empty:
    raise ValueError(
        "No valid cell_line/count_error rows were found in the CSV."
    )


# ============================================================
# SORT CELL TYPES BY MEDIAN COUNT ERROR
# ============================================================

median_errors = (
    plot_df.groupby("cell_line")["count_error"]
    .median()
    .sort_values()
)

cell_type_order = median_errors.index.tolist()

# Collect the count-error values for each cell type in sorted order.
data_for_boxplot = [
    plot_df.loc[
        plot_df["cell_line"] == cell_type,
        "count_error"
    ].values
    for cell_type in cell_type_order
]

# Number of evaluated images per cell type.
sample_sizes = (
    plot_df.groupby("cell_line")
    .size()
    .reindex(cell_type_order)
)


# ============================================================
# DISPLAY LABELS
# ============================================================
# Format:
#     Tissue of origin
#     (Cell type / disease designation)
#
# The key must match the value in the CSV "cell_line" column.

# ============================================================
# DISPLAY LABELS
# ============================================================
# Format:
#     Tissue of origin
#     (LIVECell cell-line identifier)

CELL_TYPE_LABELS = {
    "A172": "Brain",
    "BT474": "Breast",
    "BV2": "Brain",
    "Huh7": "Liver",
    "MCF7": "Breast",
    "SHSY5Y": "Bone marrow",
    "SkBr3": "Breast",
    "SKOV3": "Ovary"
}

# Tissue on the first line; LIVECell cell type/line in parentheses below.
# The sample size is displayed on the final line.
x_labels = [
    f"{CELL_TYPE_LABELS.get(cell_type, 'Unknown tissue')}\n"
    f"({cell_type})\n"
    f"(n={sample_sizes[cell_type]})"
    for cell_type in cell_type_order
]


# ============================================================
# CREATE BOX-AND-WHISKER PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(14, 8))

ax.boxplot(
    data_for_boxplot,
    tick_labels=x_labels,
    patch_artist=True,
    showfliers=True,
    medianprops={
        "color": "black",
        "linewidth": 2
    },
    boxprops={
        "facecolor": "#8ecae6",
        "edgecolor": "#023047",
        "linewidth": 1.5
    },
    whiskerprops={
        "color": "#023047",
        "linewidth": 1.5
    },
    capprops={
        "color": "#023047",
        "linewidth": 1.5
    },
    flierprops={
        "marker": "o",
        "markerfacecolor": "#fb8500",
        "markeredgecolor": "#fb8500",
        "markersize": 4,
        "alpha": 0.60
    }
)

# Horizontal zero-error reference line:
# count_error = predicted_cell_count - ground_truth_cell_count
ax.axhline(
    y=0,
    color="red",
    linestyle="--",
    linewidth=1.5,
    label="No count error"
)

ax.set_title(
    "KNN Cell-Count Error by Tissue of Origin and Cell Type\n"
    "(Sorted by Median Count Error)",
    fontsize=15,
    fontweight="bold"
)

ax.set_xlabel(
    "Tissue of Origin (Cell Type)",
    fontsize=12,
    labelpad=14
)

ax.set_ylabel(
    "Count Error\n(Predicted Cell Count − Ground-Truth Cell Count)",
    fontsize=12
)

ax.tick_params(
    axis="x",
    labelsize=10,
    pad=8
)

ax.grid(
    axis="y",
    linestyle=":",
    linewidth=1,
    alpha=0.65
)

ax.legend(
    loc="best",
    frameon=True
)

plt.tight_layout()

plt.savefig(
    PLOT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"Saved box-and-whisker plot to:\n{PLOT_FILE}")


# ============================================================
# SAVE A SUMMARY TABLE
# ============================================================

summary = (
    plot_df.groupby("cell_line")["count_error"]
    .agg(
        number_of_images="count",
        mean_count_error="mean",
        median_count_error="median",
        standard_deviation="std",
        minimum_count_error="min",
        maximum_count_error="max"
    )
    .sort_values("median_count_error")
)

# Add readable tissue/cell-type labels to the summary CSV.
summary.insert(
    0,
    "tissue_and_cell_type",
    [
        CELL_TYPE_LABELS.get(cell_type, cell_type)
        .replace("\n", " ")
        for cell_type in summary.index
    ]
)

summary.to_csv(SUMMARY_FILE)

print(f"\nSaved summary table to:\n{SUMMARY_FILE}")
print("\nCount-error summary by cell type:")
print(summary.round(2))