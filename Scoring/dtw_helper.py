import numpy as np
import dtaidistance
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


class DTWHelper:
    """Helper class for Dynamic Time Warping computations."""

    def __init__(self, method: str = 'fastdtw'):
        self.method = method

    def compute_similarity(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """Compute DTW similarity between two sequences."""
        if self.method == 'fastdtw':
            return DTWHelper.compute_similarity_fastdtw(seq1, seq2)
        elif self.method == 'dtaidistance':
            return DTWHelper.compute_similarity_dtaidistance(seq1, seq2)
        elif self.method == 'dtaidistance_fast':
            return DTWHelper.compute_similarity_dtaidistance_fast(seq1, seq2)
        else:
            raise ValueError(f"Unknown DTW method: {self.method}")

    @staticmethod
    def compute_similarity_fastdtw(seq1: np.ndarray, seq2: np.ndarray) -> float:
        """Compute DTW similarity between two sequences using FastDTW."""
        distance, _ = fastdtw(seq1, seq2, dist=euclidean)
        normalized_distance = distance / (len(seq2) + len(seq1))
        return 1 / (1 + normalized_distance)

    @staticmethod
    def compute_similarity_dtaidistance(seq1: np.ndarray, seq2: np.ndarray) -> float:
        """Compute DTW similarity between two sequences using dtaidistance."""
        distance = dtaidistance.dtw.distance(seq1, seq2)
        normalized_distance = distance / (len(seq2) + len(seq1))
        return 1 / (1 + normalized_distance)

    @staticmethod
    def compute_similarity_dtaidistance_fast(seq1: np.ndarray, seq2: np.ndarray) -> float:
        """Compute DTW similarity between two sequences using dtaidistance (fast approximation)."""
        # Convert input arrays to numpy.float64
        seq1 = seq1.astype(np.float64)
        seq2 = seq2.astype(np.float64)

        options = {
            'window': 10,  # Example window size; adjust as needed
            'use_c': True  # Enable the C library, which is faster
        }
        distance = dtaidistance.dtw.distance(seq1, seq2, **options)
        normalized_distance = distance / (len(seq2) + len(seq1))
        return 1 / (1 + normalized_distance)
