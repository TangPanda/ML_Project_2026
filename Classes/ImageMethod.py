import numpy as np
from scipy.ndimage import uniform_filter
from skimage.filters import sobel, gabor
from skimage.filters.rank import entropy
from skimage.morphology import footprint_rectangle
from scipy.ndimage import gaussian_filter

class ImageMethod:
    """Base class for image-processing methods."""

    def __init__(self, image):
        self.image = np.asarray(image, dtype=np.float32)
        self.result = None

    def show_result(self):
        if self.result is None:
            raise ValueError("Run the processing method first.")

        return self.result


class GaussianSmoothing(ImageMethod):
    """Smooth the image to reduce noise."""

    def apply(self, sigma, radius=1.0):
        self.result = gaussian_filter(self.image, sigma=sigma, radius=radius)
        return self.result


class BackgroundSubtraction(ImageMethod):
    """Subtract a broad background estimate to emphasize bright structures."""
    def apply(self, sigma=8.0):
        background = gaussian_filter(self.image, sigma=sigma)
        self.background = background
        self.result = np.clip(self.image - background, 0, None)
        return self.result


class SobelEdges(ImageMethod):
    """Detect edges using image intensity gradients."""
    def apply(self):
        self.result = sobel(self.image)
        return self.result


class GaborFilter(ImageMethod):
    """Detect oriented, line-like image structures."""
    def apply(self, frequency=0.1, theta=0):
        real, imag = gabor(
            self.image,
            frequency=frequency,
            theta=theta
        )
        self.result = np.sqrt(real**2 + imag**2)
        return self.result

class LocalVariance(ImageMethod):
    def apply(self, window_size=9):
        image = self.image
        local_mean = uniform_filter(image, size=window_size)
        local_mean_squared = uniform_filter(image**2, size=window_size)

        self.result = np.maximum(
            local_mean_squared - local_mean**2, 0
        )
        return self.result


class LocalStandardDeviation(ImageMethod):
    def apply(self, window_size=9):
        image = self.image
        local_mean = uniform_filter(image, size=window_size)
        local_mean_squared = uniform_filter(image**2, size=window_size)

        local_variance = np.maximum(
            local_mean_squared - local_mean**2, 0
        )
        self.result = np.sqrt(local_variance)
        return self.result


class LocalEntropy(ImageMethod):
    def apply(self, window_size=9, n_bins=32):
        # Entropy requires a nonnegative integer image.
        image = np.clip(self.image, 0, 1)
        image_uint = (image * (n_bins - 1)).astype(np.uint8)

        footprint = footprint_rectangle(
            shape=(window_size, window_size)
        )

        self.result = entropy(image_uint, footprint=footprint)
        return self.result