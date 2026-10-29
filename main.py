from PIL import Image
import numpy as np
import tifffile as tiff
from readlif.reader import LifFile
from Classes.LifImage import LifImage
from Classes.singleSlice import singleSlice
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from Classes.clustering import clustering



new_file = LifFile("C:\\Users\\dance\\OneDrive\\Documents\\Python\\ML_Project_2026\\Images\\20260923-cldn-lyn2-GFP-col28a2a-lyn2-mKate-31-33-hpf.lif")

# Check how many image series are inside the project
print(f"Total image series: {new_file.num_images}")
all_images_1t = [image for image in new_file.get_iter_image() if (image.dims.z > 1 and image.dims.t == 1)]
image1 = all_images_1t[0]

lif1 = LifImage(image1)
lif1zs = lif1.zframes()

channel_results = {}

for channel in ["channel_0", "channel_1", "merged_channels"]:

    # Create a DataFrame for this channel
    df = pd.DataFrame({
        "slice number": [
            f"image {i}" for i in range(len(lif1zs))
        ]
    })

    # Find the maximum intensity across the entire Z-stack
    max_intensity = max(
        np.max(np.asarray(matrix))
        for matrix in lif1zs[channel]
    )

    if max_intensity == 0:
        raise ValueError(f"Maximum intensity for {channel} is zero.")

    scaled_vals = []
    histograms = []

    for i in range(len(lif1zs)):
        slice_obj = singleSlice(lif1zs.loc[i, :])

        # Scale intensities using the stack-wide maximum
        scaled_slice = slice_obj.scale_vals(
            index=channel,
            scaleBy=max_intensity
        )

        # Calculate histogram of scaled intensities
        hist = slice_obj.histogram(bins=256)

        scaled_vals.append(scaled_slice)
        histograms.append(hist)

    # Store results
    df["scaled matrix"] = scaled_vals
    df["histogram"] = histograms
    channel_results[channel] = df
# Select slice 50 and channel 0
slice50 = singleSlice(lif1zs.loc[50])

# Use the already-scaled image from channel_results
slice50.scaled_data = channel_results["channel_0"].loc[50, "scaled matrix"]

# Apply all four image-processing methods
results = slice50.apply_image_methods_localN()

# Access individual results
smoothed = results["gaussian"]
background_subtracted = results["background_subtraction"]
edges = results["sobel_edges"]
gabor_response = results["gabor"]

images = [
    (slice50.scaled_data, "Original scaled image"),
    (smoothed, "Gaussian smoothing"),
    (background_subtracted, "Background subtraction"),
    (edges, "Sobel edges"),
    (gabor_response, "Gabor response")
]


texture_results_50 = slice50.apply_image_methods_texture(
    window_size=9,
    entropy_bins=32
)

# Extract each result
local_variance_50 = texture_results_50["local_variance"]
local_std_50 = texture_results_50["local_std"]
local_entropy_50 = texture_results_50["local_entropy"]
fig, axes = plt.subplots(1, 3, figsize=(15, 7))

images = [
    (local_variance_50, "Local Variance"),
    (local_std_50, "Local Standard Deviation"),
    (local_entropy_50, "Local Entropy")
]

image_results = slice50.image_method_results

texture_images = [
    (local_variance_50, "Local Variance"),
    (local_std_50, "Local Standard Deviation"),
    (local_entropy_50, "Local Entropy")
]


pixel_coordinates = slice50.normalize_coordinates()
plt.figure(figsize=(7, 6))
plt.scatter(
    pixel_coordinates["x_norm"],
    pixel_coordinates["y_norm"],
    c=pixel_coordinates["intensity"],
    s=1,
    cmap="gray"
)


# Display histograms for slice 50
# fig, axes = plt.subplots(1, 3, figsize=(15, 6))
# for ax, channel in zip(axes, ["channel_0", "channel_1", "merged_channels"]):
#     hist = channel_results[channel].loc[50, "histogram"]
#
#     counts = hist["counts"]
#     bin_edges = hist["bin_edges"]
#     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
#
#     ax.plot(bin_centers, np.log1p(counts))
#     ax.set_xlabel("Normalized pixel intensity")
#     ax.set_ylabel("log(Count + 1)")
#     ax.set_title(f"{channel} — Z slice 50")
# plt.tight_layout()
# plt.show()
#
# # Display Z slice 50 for  channels
# fig, axes = plt.subplots(1, 3, figsize=(18, 8))
#
# for ax, channel in zip(axes, ["channel_0", "channel_1", "merged_channels"]):
#     scaled_img = channel_results[channel].loc[50, "scaled matrix"]
#     ax.imshow(scaled_img, cmap="gray", vmin=0, vmax=1)
#     ax.set_title(f"Z slice 50 — {channel}")
#     ax.axis("off")
# plt.tight_layout()
# plt.show()



X = pd.DataFrame({
    "x_norm": pixel_coordinates["x_norm"].to_numpy(),
    "y_norm": pixel_coordinates["y_norm"].to_numpy(),
    "intensity": pixel_coordinates["intensity"].to_numpy(),
    "gaussian": np.asarray(smoothed).ravel(),
    "background_subtraction": np.asarray(background_subtracted).ravel(),
    "sobel": np.asarray(edges).ravel(),
    "gabor": np.asarray(gabor_response).ravel(),
    "variance": np.asarray(local_variance_50).ravel(),
    "standard_deviation": np.asarray(local_std_50).ravel(),
    "entropy": np.asarray(local_entropy_50).ravel()
})

# Convert to NumPy array for SVM
X = X.to_numpy(dtype=np.float32)

print("Feature matrix shape:", X.shape)

# Initialize the clustering model
cluster_model = clustering(n_clusters=2)

# Assign every pixel to cluster 0 or cluster 1
pixel_labels = cluster_model.fit_predict(X)

# Check the labels
print("Feature matrix shape:", X.shape)
print("Pixel labels shape:", pixel_labels.shape)
print("Number of pixels in each cluster:", np.bincount(pixel_labels))

# Reshape the pixel labels into the original image dimensions
height, width = slice50.scaled_data.shape
cluster_image = pixel_labels.reshape(height, width)


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Original image
axes[0].imshow(slice50.scaled_data, cmap="gray")
axes[0].set_title("Original Image — Channel 0, Slice 50")
axes[0].axis("off")

# Clustered image: two distinct colors
im = axes[1].imshow(
    cluster_image,
    cmap="viridis",
    vmin=0,
    vmax=1,
    interpolation="nearest"
)
axes[1].set_title("K-means Clustering (2 Groups)")
axes[1].axis("off")

cbar = fig.colorbar(im, ax=axes[1], ticks=[0, 1], fraction=0.046, pad=0.04)
cbar.set_label("Cluster label")

plt.tight_layout()
plt.show()