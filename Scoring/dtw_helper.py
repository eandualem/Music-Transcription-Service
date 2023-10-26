import numpy as np
import dtaidistance
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


class DTWHelper:
    """Helper class for Dynamic Time Warping computations."""

    def __init__(self, method: str = "fastdtw"):
        self.method = method

    @staticmethod
    def tolerance_euclidean(x, y, tolerance: float) -> float:
        """Custom Euclidean distance function with an optional tolerance."""
        return np.clip(euclidean(x, y), 0, tolerance)

    @staticmethod
    def compute_normalized_distance(distance: float, total_length: int) -> float:
        """Compute the normalized distance and return the similarity score."""
        normalized_distance = distance / total_length
        return 1 / (1 + normalized_distance)

    def compute_similarity(self, seq1: np.ndarray, seq2: np.ndarray, tolerance: float = None) -> float:
        """Compute DTW similarity between two sequences."""

        if seq1.size == 0 or seq2.size == 0:
            return "Error: Input sequences must not be empty"

        if not np.all(np.isfinite(seq1)) or not np.all(np.isfinite(seq2)):
            return "Error: Input sequences must contain only finite values"

        if seq1.ndim != seq2.ndim:
            return "Error: Input sequences must have matching dimensions"

        seq1 = seq1.reshape(-1, 1)
        seq2 = seq2.reshape(-1, 1)
        total_length = len(seq2) + len(seq1)

        if self.method == "fastdtw":
            return self.compute_similarity_fastdtw(seq1, seq2, tolerance, total_length)
        elif self.method == "dtaidistance_fast":
            return self.compute_similarity_dtaidistance_fast(seq1, seq2, tolerance, total_length)
        else:
            raise ValueError(f"🚨 Unknown DTW method: {self.method}")

    @staticmethod
    def compute_similarity_fastdtw(seq1: np.ndarray, seq2: np.ndarray, tolerance: float, total_length: int) -> float:
        """Compute DTW similarity between two sequences using FastDTW."""

        distance, _ = fastdtw(seq1, seq2, dist=lambda x, y: DTWHelper.tolerance_euclidean(x, y, tolerance))
        return DTWHelper.compute_normalized_distance(distance, total_length)

    @staticmethod
    def compute_similarity_dtaidistance_fast(
        seq1: np.ndarray, seq2: np.ndarray, tolerance: float, total_length: int
    ) -> float:
        """Compute DTW similarity between two sequences using dtaidistance (fast approximation)."""
        seq1 = seq1.astype(np.float64)
        seq2 = seq2.astype(np.float64)

        options = {"window": int(max(len(seq1), len(seq2)) * tolerance), "use_c": True}
        distance = dtaidistance.dtw.distance(seq1, seq2, **options)
        return DTWHelper.compute_normalized_distance(distance, total_length)
