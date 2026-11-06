from pathlib import Path
import json
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
import time
import os
import psutil

from pycocotools.coco import COCO
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from skimage.filters import gaussian, sobel
from skimage.exposure import rescale_intensity
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed, find_boundaries
from scipy import ndimage as ndi

# ============================================================
# USER SETTINGS
# ============================================================

LIVE_ROOT = Path(
    r"C:\Users\abreun01\OneDrive - NYU Langone Health\MSCI\FALL 2026\Machine Learning\Project\LIVECell"
)

ANNOTATIONS_DIR = LIVE_ROOT / "Annotations"

# Used only for constructing the KNN training data.
TRAIN_ANNOTATION_FILE = (
    ANNOTATIONS_DIR / "livecell_coco_train.json"
)

# Used only for test-image evaluation and ground-truth comparison.
TEST_ANNOTATION_FILE = (
    ANNOTATIONS_DIR / "livecell_coco_test.json"
)

# The script searches this directory and all subdirectories
# to locate the TIFF files named in the COCO annotation JSONs.
IMAGE_DIR = LIVE_ROOT / "images"

# Output folder for labels, outlines, metrics, and cost logs.
OUTPUT_DIR = LIVE_ROOT / "knn_test_segmentation_output"

PIXELS_PER_CLASS = 40000
N_NEIGHBORS = 7
MIN_CELL_AREA = 100
WATERSHED_MIN_DISTANCE = 12
RANDOM_SEED = 42

# ============================================================
# IMAGE / FEATURE FUNCTIONS
# ============================================================

def read_grayscale_image(path):
    """Read TIFF and convert it to a normalized 2-D grayscale image."""
    image = tiff.imread(path)
    image = np.squeeze(image)

    # RGB/RGBA image
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        image = image[..., :3].mean(axis=-1)

    # For a multi-plane stack, use maximum projection.
    elif image.ndim == 3:
        image = np.max(image, axis=0)

    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D image after conversion; got {image.shape} for {path}")

    image = image.astype(np.float32)

    # Robust normalization helps reduce microscope-to-microscope intensity variation.
    low, high = np.percentile(image, (1, 99))
    if high > low:
        image = np.clip((image - low) / (high - low), 0, 1)
    else:
        image = np.zeros_like(image, dtype=np.float32)

    return image


def make_features(image):
    """
    Create a feature vector at every pixel.

    Features:
    - normalized phase-contrast intensity
    - Gaussian-blurred intensity, sigma 1
    - Gaussian-blurred intensity, sigma 3
    - Sobel edge magnitude
    """
    smooth_1 = gaussian(image, sigma=1)
    smooth_3 = gaussian(image, sigma=3)
    edges = sobel(image)

    features = np.stack(
        [
            image,
            smooth_1,
            smooth_3,
            edges
        ],
        axis=-1
    )

    return features.reshape(-1, features.shape[-1])


def locate_image(image_directory, file_name):
    """Find an annotation-referenced image inside the training image directory."""
    direct_path = image_directory / file_name
    if direct_path.exists():
        return direct_path

    # Handles COCO filenames that contain no subfolder or unexpected subfolders.
    matches = list(image_directory.rglob(Path(file_name).name))
    if matches:
        return matches[0]

    return None

def create_ground_truth_labels(coco, image_id, image_shape):
    """
    Create an integer instance-label image from LIVECell COCO annotations.

    Label meanings:
        0 = background
        1, 2, 3, ... = individual annotated cells
    """
    ground_truth_labels = np.zeros(image_shape, dtype=np.int32)

    annotation_ids = coco.getAnnIds(imgIds=[image_id], iscrowd=None)
    annotations = coco.loadAnns(annotation_ids)

    label_number = 1

    for ann in annotations:
        # Convert the COCO polygon/RLE annotation to a binary mask.
        cell_mask = coco.annToMask(ann).astype(bool)

        # Ensure the annotation and TIFF image have matching dimensions.
        if cell_mask.shape != ground_truth_labels.shape:
            print(
                f"    Warning: annotation mask shape {cell_mask.shape} "
                f"does not match image shape {ground_truth_labels.shape}; "
                f"skipping one annotation."
            )
            continue

        # Preserve the first cell label at pixels where annotations overlap.
        available_pixels = cell_mask & (ground_truth_labels == 0)
        ground_truth_labels[available_pixels] = label_number

        label_number += 1

    return ground_truth_labels

def calculate_evaluation_metrics(predicted_labels, ground_truth_labels):
    """
    Compare predicted instance labels with LIVECell COCO ground truth.

    Metrics:
    - predicted_cell_count
    - ground_truth_cell_count
    - count error
    - absolute and percent count error
    - foreground IoU: pixel-level overlap of cell vs. background masks

    Note:
    This is foreground/background IoU, not one-to-one instance-matching IoU.
    """

    # Convert the instance-label images to binary foreground masks.
    predicted_foreground = predicted_labels > 0
    ground_truth_foreground = ground_truth_labels > 0

    # Pixel-level foreground intersection-over-union.
    intersection = np.logical_and(
        predicted_foreground,
        ground_truth_foreground
    ).sum()

    union = np.logical_or(
        predicted_foreground,
        ground_truth_foreground
    ).sum()

    foreground_iou = intersection / union if union > 0 else np.nan

    # Because labels are sequentially assigned, max label = number of cells.
    predicted_count = int(predicted_labels.max())
    ground_truth_count = int(ground_truth_labels.max())

    count_error = predicted_count - ground_truth_count
    absolute_count_error = abs(count_error)

    if ground_truth_count > 0:
        percent_count_error = (
            100 * absolute_count_error / ground_truth_count
        )
    else:
        percent_count_error = np.nan

    return {
        "predicted_cell_count": predicted_count,
        "ground_truth_cell_count": ground_truth_count,
        "count_error": count_error,
        "absolute_count_error": absolute_count_error,
        "percent_count_error": percent_count_error,
        "foreground_iou": foreground_iou
    }

def get_process_memory_mb():
    """
    Return the current Python process memory use in megabytes.

    RSS = resident set size, meaning physical RAM currently used
    by this Python process.
    """
    process = psutil.Process(os.getpid())
    memory_bytes = process.memory_info().rss
    return memory_bytes / (1024 ** 2)


def format_elapsed_time(seconds):
    """Format elapsed time as HH:MM:SS."""
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

# ============================================================
# BUILD TRAINING DATA FROM LIVECell COCO ANNOTATIONS
# ============================================================

def build_training_pixels(coco, train_image_dir, pixels_per_class=40000):
    """
    Produce balanced pixel samples:
      0 = background
      1 = cell foreground
    """
    rng = np.random.default_rng(RANDOM_SEED)

    foreground_features = []
    background_features = []

    image_ids = coco.getImgIds()
    print(f"Preparing training data from {len(image_ids)} annotated images...")

    for idx, image_id in enumerate(image_ids, start=1):
        image_info = coco.loadImgs(image_id)[0]
        image_path = locate_image(train_image_dir, image_info["file_name"])

        if image_path is None:
            print(f"Skipping missing file: {image_info['file_name']}")
            continue

        try:
            image = read_grayscale_image(image_path)
        except Exception as error:
            print(f"Skipping unreadable image {image_path.name}: {error}")
            continue

        # Build one binary foreground mask from all annotated cells.
        cell_mask = np.zeros(image.shape, dtype=bool)
        annotation_ids = coco.getAnnIds(imgIds=[image_id], iscrowd=None)

        for ann in coco.loadAnns(annotation_ids):
            ann_mask = coco.annToMask(ann).astype(bool)

            # Defensive handling if annotation/image dimensions do not match.
            if ann_mask.shape == cell_mask.shape:
                cell_mask |= ann_mask

        features = make_features(image)

        foreground_indices = np.flatnonzero(cell_mask.ravel())
        background_indices = np.flatnonzero(~cell_mask.ravel())

        # Sample a limited number of pixels from each class.
        n_fg = min(300, len(foreground_indices))
        n_bg = min(300, len(background_indices))

        if n_fg > 0:
            selected_fg = rng.choice(foreground_indices, size=n_fg, replace=False)
            foreground_features.append(features[selected_fg])

        if n_bg > 0:
            selected_bg = rng.choice(background_indices, size=n_bg, replace=False)
            background_features.append(features[selected_bg])

        if idx % 100 == 0:
            print(f"Processed {idx}/{len(image_ids)} training images")

    X_fg = np.vstack(foreground_features)
    X_bg = np.vstack(background_features)

    # Balance the classes for KNN.
    n = min(pixels_per_class, len(X_fg), len(X_bg))
    X_fg = X_fg[rng.choice(len(X_fg), n, replace=False)]
    X_bg = X_bg[rng.choice(len(X_bg), n, replace=False)]

    X = np.vstack([X_fg, X_bg])
    y = np.concatenate([
        np.ones(len(X_fg), dtype=np.uint8),
        np.zeros(len(X_bg), dtype=np.uint8)
    ])

    # Shuffle training pixels.
    order = rng.permutation(len(y))
    X, y = X[order], y[order]

    print(f"Final KNN training set: {len(y):,} pixels")
    print(f"Foreground pixels: {(y == 1).sum():,}")
    print(f"Background pixels: {(y == 0).sum():,}")

    return X, y


# ============================================================
# INSTANCE SEGMENTATION AND OUTPUT
# ============================================================

def predict_in_chunks(model, features, chunk_size=50000):
    """Avoid creating excessively large KNN calculations at once."""
    predictions = np.empty(len(features), dtype=np.uint8)

    for start in range(0, len(features), chunk_size):
        end = min(start + chunk_size, len(features))
        predictions[start:end] = model.predict(features[start:end])

    return predictions


def separate_cells(binary_mask):
    """
    Convert a binary foreground mask into individual cell labels
    using distance-transform watershed.
    """

    # Remove objects smaller than MIN_CELL_AREA pixels.
    # max_size replaces the deprecated min_size argument in newer scikit-image.
    binary_mask = remove_small_objects(
        binary_mask.astype(bool),
        max_size=MIN_CELL_AREA - 1
    )

    distance = ndi.distance_transform_edt(binary_mask)

    coordinates = peak_local_max(
        distance,
        min_distance=WATERSHED_MIN_DISTANCE,
        labels=binary_mask
    )

    markers = np.zeros_like(binary_mask, dtype=np.int32)
    if len(coordinates) > 0:
        markers[tuple(coordinates.T)] = np.arange(1, len(coordinates) + 1)

    markers = label(markers > 0)

    if markers.max() == 0:
        return np.zeros_like(binary_mask, dtype=np.int32)

    labels = watershed(
        -distance,
        markers,
        mask=binary_mask
    )

    # Remove small watershed fragments, then relabel sequentially.
    valid_labels = np.zeros_like(labels, dtype=np.int32)
    new_label = 1

    for region in regionprops(labels):
        if region.area >= MIN_CELL_AREA:
            valid_labels[labels == region.label] = new_label
            new_label += 1

    return valid_labels


def save_results(image_path, image, labels, output_dir):
    """Save labeled TIFF, outline PNG, and return cell count."""
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = image_path.stem
    boundaries = find_boundaries(labels, mode="outer")
    cell_count = int(labels.max())

    # Save integer instance-label mask.
    tiff.imwrite(
        output_dir / f"{stem}_instance_labels.tif",
        labels.astype(np.uint16)
    )

    # Save an image with red outlines overlaid on the phase image.
    rgb = np.dstack([image, image, image])
    rgb[boundaries] = [1.0, 0.0, 0.0]

    plt.imsave(output_dir / f"{stem}_outlines.png", rgb)

    return cell_count


# ============================================================
# MAIN
# ============================================================
def main():
    # Start timing the full program.
    total_start_time = time.perf_counter()
    starting_memory_mb = get_process_memory_mb()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    computational_log_path = OUTPUT_DIR / "computational_cost_log.txt"

    # Open a text log file. "w" overwrites an older log from a previous run.
    with open(computational_log_path, "w", encoding="utf-8") as log_file:

        def log(message):
            """Print a message and also write it to the log file."""
            print(message)
            log_file.write(message + "\n")
            log_file.flush()

        # --------------------------------------------------------
        # Record settings and initial computational information
        # --------------------------------------------------------
        log("=" * 70)
        log("KNN LIVECell Segmentation: Computational Cost Log")
        log("=" * 70)
        log(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        log(f"Initial process memory: {starting_memory_mb:.2f} MB")
        log(f"PIXELS_PER_CLASS: {PIXELS_PER_CLASS:,}")
        log(f"N_NEIGHBORS: {N_NEIGHBORS}")
        log(f"MIN_CELL_AREA: {MIN_CELL_AREA}")
        log(f"WATERSHED_MIN_DISTANCE: {WATERSHED_MIN_DISTANCE}")
        log(f"Training annotation file: {TRAIN_ANNOTATION_FILE}")
        log(f"Test annotation file: {TEST_ANNOTATION_FILE}")
        log(f"Image directory: {IMAGE_DIR}")
        log(f"Output directory: {OUTPUT_DIR}")
        log("")

        # --------------------------------------------------------
        # Check required paths
        # --------------------------------------------------------
        if not TRAIN_ANNOTATION_FILE.exists():
            raise FileNotFoundError(
                f"Could not find training annotation file:\n"
                f"{TRAIN_ANNOTATION_FILE}"
            )

        if not TEST_ANNOTATION_FILE.exists():
            raise FileNotFoundError(
                f"Could not find test annotation file:\n"
                f"{TEST_ANNOTATION_FILE}"
            )

        if not IMAGE_DIR.exists():
            raise FileNotFoundError(
                f"Could not find image directory:\n{IMAGE_DIR}"
            )

        # --------------------------------------------------------
        # Load annotations
        # --------------------------------------------------------
        annotation_load_start = time.perf_counter()

        log("Loading LIVECell TRAIN COCO annotations...")
        train_coco = COCO(str(TRAIN_ANNOTATION_FILE))

        log("Loading LIVECell TEST COCO annotations...")
        test_coco = COCO(str(TEST_ANNOTATION_FILE))

        annotation_load_seconds = time.perf_counter() - annotation_load_start

        log(
            f"Annotation loading time: "
            f"{annotation_load_seconds:.2f} seconds "
            f"({format_elapsed_time(annotation_load_seconds)})"
        )
        log(f"Memory after annotation loading: {get_process_memory_mb():.2f} MB")
        log("")

        # --------------------------------------------------------
        # Build training pixels
        # --------------------------------------------------------
        training_data_start = time.perf_counter()

        log("Building KNN pixel training data from TRAIN images...")

        X_train, y_train = build_training_pixels(
            train_coco,
            IMAGE_DIR,
            pixels_per_class=PIXELS_PER_CLASS
        )

        training_data_seconds = time.perf_counter() - training_data_start
        training_data_memory_mb = get_process_memory_mb()

        log("")
        log(
            f"Training-data preparation time: "
            f"{training_data_seconds:.2f} seconds "
            f"({format_elapsed_time(training_data_seconds)})"
        )
        log(f"Training feature matrix shape: {X_train.shape}")
        log(f"Training label vector shape: {y_train.shape}")
        log(f"Memory after training-data preparation: {training_data_memory_mb:.2f} MB")
        log("")

        # --------------------------------------------------------
        # Fit KNN model
        # --------------------------------------------------------
        knn_fit_start = time.perf_counter()

        log("Training KNN classifier...")

        model = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(
                n_neighbors=N_NEIGHBORS,
                weights="distance",
                n_jobs=-1
            )
        )

        model.fit(X_train, y_train)

        knn_fit_seconds = time.perf_counter() - knn_fit_start
        knn_fit_memory_mb = get_process_memory_mb()

        log("KNN training complete.")
        log(
            f"KNN fitting time: {knn_fit_seconds:.2f} seconds "
            f"({format_elapsed_time(knn_fit_seconds)})"
        )
        log(f"Memory after KNN fitting: {knn_fit_memory_mb:.2f} MB")
        log("")

        # --------------------------------------------------------
        # Evaluate test images
        # --------------------------------------------------------
        test_image_ids = test_coco.getImgIds()

        log(
            f"Segmenting and evaluating "
            f"{len(test_image_ids):,} LIVECell TEST image(s)..."
        )
        log("")

        results = []

        test_evaluation_start = time.perf_counter()

        for number, image_id in enumerate(test_image_ids, start=1):
            image_start_time = time.perf_counter()
            memory_before_image_mb = get_process_memory_mb()

            image_info = test_coco.loadImgs(image_id)[0]
            image_name = image_info["file_name"]

            image_path = locate_image(IMAGE_DIR, image_name)

            if image_path is None:
                log(
                    f"[{number}/{len(test_image_ids)}] "
                    f"Skipping missing image: {image_name}"
                )
                continue

            log(f"[{number}/{len(test_image_ids)}] {image_name}")

            try:
                image = read_grayscale_image(image_path)
            except Exception as error:
                log(f"    Skipping unreadable image: {error}")
                continue

            # KNN foreground/background prediction.
            features = make_features(image)
            foreground_prediction = predict_in_chunks(model, features)
            binary_mask = foreground_prediction.reshape(image.shape).astype(bool)

            # Watershed cell-instance separation.
            predicted_labels = separate_cells(binary_mask)

            # Ground-truth labels from test COCO annotation.
            ground_truth_labels = create_ground_truth_labels(
                test_coco,
                image_id,
                image.shape
            )

            # Calculate count and pixel-level overlap metrics.
            metrics = calculate_evaluation_metrics(
                predicted_labels,
                ground_truth_labels
            )

            # Save predicted labels and predicted red-outline image.
            save_results(
                image_path=image_path,
                image=image,
                labels=predicted_labels,
                output_dir=OUTPUT_DIR
            )

            # Save integer ground-truth instance labels.
            tiff.imwrite(
                OUTPUT_DIR / f"{image_path.stem}_ground_truth_labels.tif",
                ground_truth_labels.astype(np.uint16)
            )

            # Green = LIVECell ground truth.
            gt_boundaries = find_boundaries(
                ground_truth_labels,
                mode="outer"
            )

            gt_rgb = np.dstack([image, image, image])
            gt_rgb[gt_boundaries] = [0.0, 1.0, 0.0]

            plt.imsave(
                OUTPUT_DIR / f"{image_path.stem}_ground_truth_outlines.png",
                gt_rgb
            )

            # Green = ground truth; red = KNN prediction.
            predicted_boundaries = find_boundaries(
                predicted_labels,
                mode="outer"
            )

            comparison_rgb = np.dstack([image, image, image])
            comparison_rgb[gt_boundaries] = [0.0, 1.0, 0.0]
            comparison_rgb[predicted_boundaries] = [1.0, 0.0, 0.0]

            plt.imsave(
                OUTPUT_DIR / f"{image_path.stem}_comparison.png",
                comparison_rgb
            )

            # Per-image computational costs.
            image_runtime_seconds = time.perf_counter() - image_start_time
            memory_after_image_mb = get_process_memory_mb()
            memory_change_mb = memory_after_image_mb - memory_before_image_mb

            results.append({
                "image_id": image_id,
                "image_file": image_name,
                "cell_line": image_name.split("_")[0],

                "image_height_pixels": image.shape[0],
                "image_width_pixels": image.shape[1],
                "total_image_pixels": image.size,

                "predicted_cell_count": metrics["predicted_cell_count"],
                "ground_truth_cell_count": metrics["ground_truth_cell_count"],
                "count_error": metrics["count_error"],
                "absolute_count_error": metrics["absolute_count_error"],
                "percent_count_error": metrics["percent_count_error"],
                "foreground_iou": metrics["foreground_iou"],

                "runtime_seconds": image_runtime_seconds,
                "runtime_minutes": image_runtime_seconds / 60,
                "memory_before_mb": memory_before_image_mb,
                "memory_after_mb": memory_after_image_mb,
                "memory_change_mb": memory_change_mb,

                "predicted_label_file": (
                    f"{image_path.stem}_instance_labels.tif"
                ),
                "ground_truth_label_file": (
                    f"{image_path.stem}_ground_truth_labels.tif"
                ),
                "comparison_image_file": (
                    f"{image_path.stem}_comparison.png"
                )
            })

            log(
                f"    Predicted: {metrics['predicted_cell_count']} | "
                f"Ground truth: {metrics['ground_truth_cell_count']} | "
                f"Absolute count error: {metrics['absolute_count_error']} | "
                f"IoU: {metrics['foreground_iou']:.3f} | "
                f"Runtime: {image_runtime_seconds:.2f} s | "
                f"RAM: {memory_after_image_mb:.1f} MB"
            )

        test_evaluation_seconds = time.perf_counter() - test_evaluation_start

        # --------------------------------------------------------
        # Save per-image results
        # --------------------------------------------------------
        results_df = pd.DataFrame(results)

        csv_path = OUTPUT_DIR / "test_evaluation_results.csv"
        results_df.to_csv(csv_path, index=False)

        # --------------------------------------------------------
        # Create and save final summary
        # --------------------------------------------------------
        total_runtime_seconds = time.perf_counter() - total_start_time
        final_memory_mb = get_process_memory_mb()

        if not results_df.empty:
            summary = pd.DataFrame([{
                "number_of_test_images_evaluated": len(results_df),

                "total_predicted_cells": int(
                    results_df["predicted_cell_count"].sum()
                ),
                "total_ground_truth_cells": int(
                    results_df["ground_truth_cell_count"].sum()
                ),

                "mean_absolute_count_error": (
                    results_df["absolute_count_error"].mean()
                ),
                "mean_foreground_iou": (
                    results_df["foreground_iou"].mean()
                ),

                "mean_runtime_seconds_per_image": (
                    results_df["runtime_seconds"].mean()
                ),
                "median_runtime_seconds_per_image": (
                    results_df["runtime_seconds"].median()
                ),
                "minimum_runtime_seconds_per_image": (
                    results_df["runtime_seconds"].min()
                ),
                "maximum_runtime_seconds_per_image": (
                    results_df["runtime_seconds"].max()
                ),

                "annotation_loading_seconds": annotation_load_seconds,
                "training_data_preparation_seconds": training_data_seconds,
                "knn_fit_seconds": knn_fit_seconds,
                "test_evaluation_seconds": test_evaluation_seconds,
                "total_runtime_seconds": total_runtime_seconds,
                "total_runtime_formatted": format_elapsed_time(
                    total_runtime_seconds
                ),

                "initial_memory_mb": starting_memory_mb,
                "final_memory_mb": final_memory_mb,
                "memory_change_mb": final_memory_mb - starting_memory_mb,
                "peak_observed_memory_mb": results_df["memory_after_mb"].max()
            }])

            summary_path = OUTPUT_DIR / "test_evaluation_summary.csv"
            summary.to_csv(summary_path, index=False)

            log("")
            log("=" * 70)
            log("COMPUTATIONAL COST SUMMARY")
            log("=" * 70)
            log(f"Test images evaluated: {len(results_df):,}")
            log(
                f"Mean runtime per image: "
                f"{results_df['runtime_seconds'].mean():.2f} seconds"
            )
            log(
                f"Median runtime per image: "
                f"{results_df['runtime_seconds'].median():.2f} seconds"
            )
            log(
                f"Maximum runtime for one image: "
                f"{results_df['runtime_seconds'].max():.2f} seconds"
            )
            log(
                f"Training-data preparation: "
                f"{format_elapsed_time(training_data_seconds)}"
            )
            log(
                f"KNN fitting: "
                f"{format_elapsed_time(knn_fit_seconds)}"
            )
            log(
                f"Test evaluation: "
                f"{format_elapsed_time(test_evaluation_seconds)}"
            )
            log(
                f"Total program runtime: "
                f"{total_runtime_seconds:.2f} seconds "
                f"({format_elapsed_time(total_runtime_seconds)})"
            )
            log(f"Initial memory: {starting_memory_mb:.2f} MB")
            log(f"Final memory: {final_memory_mb:.2f} MB")
            log(
                f"Peak observed memory after an image: "
                f"{results_df['memory_after_mb'].max():.2f} MB"
            )
            log("")
            log(f"Per-image results CSV: {csv_path}")
            log(f"Summary CSV: {summary_path}")
            log(f"Text computational log: {computational_log_path}")

        else:
            log("")
            log("No test images were successfully evaluated.")
            log(
                f"Total program runtime: "
                f"{format_elapsed_time(total_runtime_seconds)}"
            )

main()