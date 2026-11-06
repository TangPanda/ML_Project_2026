from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# EDIT THESE FOLDER PATHS FOR YOUR COMPUTER
# ------------------------------------------------------------
TRAIN_VAL_IMAGES_FOLDER = Path(
    "C:/Users/abreun01/OneDrive - NYU Langone Health/MSCI/FALL 2026/Machine Learning/Project/LIVECell/images/livecell_train_val_images"
)

TEST_IMAGES_FOLDER = Path(
    "C:/Users/abreun01/OneDrive - NYU Langone Health/MSCI/FALL 2026/Machine Learning/Project/LIVECell/images/livecell_test_images"
)


# Output graph location
OUTPUT_FIGURE = Path(
    "C:/Users/abreun01/OneDrive - NYU Langone Health/MSCI/FALL 2026/Machine Learning/Project/LIVECell/images/"
)

# ------------------------------------------------------------
# LIVECell cell-line prefix -> tissue of origin
# ------------------------------------------------------------
CELL_LINE_TO_TISSUE = {
    "A172": "Brain",
    "BT474": "Breast",
    "BV2": "Brain / microglia",
    "Huh7": "Liver",
    "MCF7": "Breast",
    "SHSY5Y": "Bone marrow",
    "SKOV3": "Ovary",
    "SkBr3": "Breast",
}


def get_cell_line_from_filename(image_path):
    """
    Return the portion of the filename before the first underscore.

    Example:
        A172_Phase_A172_001.tif -> A172
        BT474_Phase_BT474_002.tif -> BT474
    """
    return image_path.stem.split("_", 1)[0]


def get_tissue_from_filename(image_path):
    """
    Extract the cell-line prefix and convert it to a tissue-of-origin label.
    """
    cell_line = get_cell_line_from_filename(image_path)

    # Keeps unexpected prefixes visible rather than silently mislabeling them.
    return CELL_LINE_TO_TISSUE.get(cell_line, f"Unknown ({cell_line})")


def count_images_by_tissue(image_folder):
    """
    Count images by tissue of origin, using the filename prefix to identify
    the LIVECell cell line and then mapping that cell line to tissue.
    """
    valid_extensions = {
        ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"
    }

    image_files = [
        file_path
        for file_path in image_folder.rglob("*")
        if file_path.is_file()
        and file_path.suffix.lower() in valid_extensions
    ]

    tissue_counts = Counter(
        get_tissue_from_filename(image_path)
        for image_path in image_files
    )

    return tissue_counts, image_files


def main():
    if not TRAIN_VAL_IMAGES_FOLDER.exists():
        raise FileNotFoundError(
            "Training/validation image folder was not found:\n"
            f"{TRAIN_VAL_IMAGES_FOLDER}"
        )

    if not TEST_IMAGES_FOLDER.exists():
        raise FileNotFoundError(
            "Test image folder was not found:\n"
            f"{TEST_IMAGES_FOLDER}"
        )

    # Count images by tissue of origin
    train_val_counts, train_val_files = count_images_by_tissue(
        TRAIN_VAL_IMAGES_FOLDER
    )
    test_counts, test_files = count_images_by_tissue(
        TEST_IMAGES_FOLDER
    )

    # Display tissues in a biologically intuitive order
    preferred_tissue_order = [
        "Brain",
        "Brain / microglia",
        "Bone marrow",
        "Breast",
        "Liver",
        "Ovary",
    ]

    all_tissues = set(train_val_counts) | set(test_counts)

    tissues = (
        [tissue for tissue in preferred_tissue_order if tissue in all_tissues]
        + sorted(
            tissue for tissue in all_tissues
            if tissue not in preferred_tissue_order
        )
    )

    train_val_values = [train_val_counts[tissue] for tissue in tissues]
    test_values = [test_counts[tissue] for tissue in tissues]

    # ------------------------------------------------------------
    # Print a summary table
    # ------------------------------------------------------------
    print("\nLIVECell image counts by tissue of origin")
    print("=" * 68)
    print(f"{'Tissue of origin':<25} {'Train/Validation':>20} {'Test':>12}")
    print("-" * 68)

    for tissue, train_count, test_count in zip(
        tissues, train_val_values, test_values
    ):
        print(
            f"{tissue:<25} "
            f"{train_count:>20,} "
            f"{test_count:>12,}"
        )

    print("-" * 68)
    print(
        f"{'Total images':<25} "
        f"{sum(train_val_values):>20,} "
        f"{sum(test_values):>12,}"
    )
    print("=" * 68)

    # ------------------------------------------------------------
    # Create grouped bar graph
    # ------------------------------------------------------------
    x_positions = np.arange(len(tissues))
    bar_width = 0.38

    fig, ax = plt.subplots(figsize=(12, 7))

    train_val_bars = ax.bar(
        x_positions - bar_width / 2,
        train_val_values,
        width=bar_width,
        color="#2E86AB",
        label="Training + validation images"
    )

    test_bars = ax.bar(
        x_positions + bar_width / 2,
        test_values,
        width=bar_width,
        color="#E45756",
        label="Test images"
    )

    ax.set_title(
        "LIVECell Images by Tissue of Origin",
        fontsize=16,
        fontweight="bold",
        pad=15
    )
    ax.set_xlabel("Tissue of origin", fontsize=12)
    ax.set_ylabel("Number of images", fontsize=12)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(tissues, rotation=30, ha="right")
    ax.legend(frameon=False)

    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # Add numerical count labels to bars
    for bars in [train_val_bars, test_bars]:
        for bar in bars:
            height = bar.get_height()

            if height > 0:
                ax.annotate(
                    f"{int(height):,}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9
                )

    plt.tight_layout()

    # Save and display graph
    OUTPUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nGraph saved to:\n{OUTPUT_FIGURE}")
    print(f"Training/validation images found: {len(train_val_files):,}")
    print(f"Test images found: {len(test_files):,}")


if __name__ == "__main__":
    main()