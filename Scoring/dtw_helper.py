import numpy as np
from fastdtw import fastdtw
from concurrent.futures import ThreadPoolExecutor, as_completed
import dtaidistance
import logging


class DTWHelper:
    """Helper class for Dynamic Time Warping computations."""

    def __init__(self, method: str = "fastdtw", parallel: bool = False):
        self.method = method
        self.parallel = parallel

    @staticmethod
    def tolerance_euclidean(x, y, tolerance: float) -> float:
        """Custom Euclidean distance function with an optional tolerance."""
        return np.clip(np.linalg.norm(x - y), 0, tolerance)

    @staticmethod
    def compute_normalized_distance(distance: float, total_length: int) -> float:
        """Compute the normalized distance and return the similarity score."""
        normalized_distance = distance / total_length
        return 1 / (1 + normalized_distance)

    def compute_similarity(self, seq1: np.ndarray, seq2: np.ndarray, tolerance: float = None) -> float:
        """Compute DTW similarity between two sequences."""
        if self.parallel:
            return self.compute_similarity_parallel(seq1, seq2, self.method_function, tolerance)
        else:
            return self.method_function(seq1, seq2, tolerance)

    def method_function(self, seq1: np.ndarray, seq2: np.ndarray, tolerance: float) -> float:
        """Determine which DTW method to use based on the 'method' attribute."""
        if self.method == "fastdtw":
            return self.compute_similarity_fastdtw(seq1, seq2, tolerance, len(seq1) + len(seq2))
        elif self.method == "dtaidistance_fast":
            return self.compute_similarity_dtaidistance_fast(seq1, seq2, tolerance, len(seq1) + len(seq2))
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
        options = {"window": int(max(len(seq1), len(seq2)) * tolerance), "use_c": True}
        distance = dtaidistance.dtw.distance(seq1, seq2, **options)
        return DTWHelper.compute_normalized_distance(distance, total_length)

    def compute_similarity_parallel(self, seq1: np.ndarray, seq2: np.ndarray, func, tolerance: float = None) -> float:
        """Compute DTW similarity in parallel over smaller chunks of audio."""
        num_chunks = 4  # Dividing the sequences into four equal parts

        # Calculating chunk size based on the longer sequence
        chunk_size = max(len(seq1), len(seq2)) // num_chunks
        # logging.info(f"chunk_size: {len(chunk_size)}")

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    func,
                    seq1[i * chunk_size : (i + 1) * chunk_size],
                    seq2[i * chunk_size : (i + 1) * chunk_size],
                    tolerance,
                )
                for i in range(num_chunks)
            ]

            scores = [future.result() for future in as_completed(futures)]

        # Averaging the scores from each chunk
        # logging.info(f"scores: {scores}")
        average_score = sum(scores) / len(scores) if scores else 0.0
        return average_score
