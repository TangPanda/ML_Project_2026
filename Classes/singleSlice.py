from scipy import stats
import pandas as pd
import numpy as np
from Classes.ImageMethod import (
    GaussianSmoothing,
    BackgroundSubtraction,
    SobelEdges,
    GaborFilter,
    LocalVariance,
    LocalStandardDeviation,
    LocalEntropy
)

class singleSlice:
    def __init__(self, data):
        self.image_data = data
        self.scaled_data = None
        self.histogram_data = None
        self.image_method_results = None
        self.texture_results = None
        self.normalized_coordinates = None


        # def __getitem__(self,index):
    #     if index not in ["channel_0", "channel_1", "merged_channels"]:
    #         raise KeyError(f"Invalid index: {index}")
    #     return self.image_data[index]

    def scale_vals(self, index, scaleBy):
        if index not in ["channel_0", "channel_1", "merged_channels"]:
            raise KeyError(f"Invalid index: {index}")
        scaled_matrix = self.image_data[index].astype('float32')/scaleBy
        self.scaled_data = scaled_matrix
        return self.scaled_data

    def histogram(self, bins=256):
        if self.scaled_data is None:
            raise ValueError("Call scale_vals() before histogram().")

        # Flatten the 2D matrix into a 1D array of pixel values
        pixel_values = self.scaled_data.flatten()

        # Calculate histogram counts and bin edges
        counts, bin_edges = np.histogram(
            pixel_values, bins=bins, range=(0, 1)
        )

        self.histogram_data = {
            "counts": counts,
            "bin_edges": bin_edges
        }

        return self.histogram_data

    def apply_image_methods_localN(
            self,
            gaussian_sigma=1.0,
            background_sigma=8.0,
            gabor_frequency=0.1,
            gabor_theta=0,
            gaussian_radius = 4
    ):
        if self.scaled_data is None:
            raise ValueError("Call scale_vals() before apply_image_methods().")

        # Apply each method to the scaled image
        gaussian = GaussianSmoothing(self.scaled_data)
        background = BackgroundSubtraction(self.scaled_data)
        sobel_edges = SobelEdges(self.scaled_data)
        gabor_filter = GaborFilter(self.scaled_data)

        # Run the methods
        results = {
            "gaussian": gaussian.apply(sigma=gaussian_sigma, radius = gaussian_radius),
            "background_subtraction": background.apply(sigma=background_sigma),
            "sobel_edges": sobel_edges.apply(),
            "gabor": gabor_filter.apply(
                frequency=gabor_frequency,
                theta=gabor_theta
            )
        }

        # Store the results for later use
        self.image_method_results = results

        return results

    def apply_image_methods_texture(
            self,
            window_size=9,
            entropy_bins=32
    ):
        if self.scaled_data is None:
            raise ValueError(
                "Call scale_vals() before apply_image_methods_texture()."
            )

        results = {
            "local_variance": LocalVariance(
                self.scaled_data
            ).apply(window_size=window_size),

            "local_std": LocalStandardDeviation(
                self.scaled_data
            ).apply(window_size=window_size),

            "local_entropy": LocalEntropy(
                self.scaled_data
            ).apply(
                window_size=window_size,
                n_bins=entropy_bins
            )
        }

        self.texture_results = results
        return results

    def normalize_coordinates(self):
        """
        Generate normalized (x, y) coordinates for every pixel
        Returns pd DataFrame:
            Original coordinates, normalized coordinates, and pixel intensity.
        # Use the scaled image if available; otherwise use the image data
        """
        if self.scaled_data is not None:
            image = np.asarray(self.scaled_data)
        else:
            raise ValueError(
                "Call scale_vals() before normalize_coordinates()."
            )

        height, width = image.shape

        # Generate pixel coordinate grids
        y, x = np.indices((height, width))

        # Normalize coordinates to [0, 1]
        x_norm = x / (width - 1) if width > 1 else np.zeros_like(x, dtype=float)
        y_norm = y / (height - 1) if height > 1 else np.zeros_like(y, dtype=float)

        # Store the coordinate grids on the object
        self.normalized_coordinates = {
            "x_norm": x_norm,
            "y_norm": y_norm
        }

        # Return one row per pixel
        return pd.DataFrame({
            "x": x.ravel(),
            "y": y.ravel(),
            "x_norm": x_norm.ravel(),
            "y_norm": y_norm.ravel(),
            "intensity": image.ravel()
        })

