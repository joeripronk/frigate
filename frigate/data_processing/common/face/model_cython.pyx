"""Cython-accelerated face recognition similarity computation.

This module provides Cython-optimized implementations of the cosine similarity
loop used in face recognition. The main hot path is comparing a single embedding
against all stored class-mean embeddings to find the best match.
"""

# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
from libc.math cimport exp

cdef inline double sigmoid(double x, double median, double range_width, double slope_factor) noexcept:
    """Map a cosine similarity value to a confidence using a sigmoid function."""
    cdef double slope = slope_factor / range_width
    return 1.0 / (1.0 + exp(-slope * (x - median)))


def classify_face(
    double[:] embedding,
    double[:, :] embeddings_matrix,
    labels: list,
    int num_embeddings,
    double median,
    double range_width,
) -> tuple:
    """Find the best matching face embedding using cosine similarity.

    Cython-optimized version of the loop in FaceNetRecognizer.classify() and
    ArcFaceRecognizer.classify() that iterates over all mean embeddings to find
    the best match.

    Args:
        embedding: 1D numpy array (embedding_dim,) - the query embedding
        embeddings_matrix: 2D numpy array (num_embeddings, embedding_dim) - class mean embeddings
        labels: Python list of label name strings (must match num_embeddings)
        num_embeddings: Number of embeddings in the matrix
        median: Sigmoid median parameter
        range_width: Sigmoid range_width parameter

    Returns:
        Tuple of (best_label: str, best_confidence: float)
    """
    cdef:
        int embedding_dim = embedding.shape[0]
        int num_labels = len(labels)
        double embedding_norm = 0.0
        double best_score = 0.0
        int best_idx = -1
        int i, j
        double dot_product
        double magnitude_B
        double cosine_similarity
        double confidence

    # Use the smaller of num_embeddings and len(labels)
    if num_labels < num_embeddings:
        num_embeddings = num_labels

    # Compute embedding norm once (it's constant across all comparisons)
    for j in range(embedding_dim):
        embedding_norm += embedding[j] * embedding[j]
    embedding_norm = embedding_norm ** 0.5

    for i in range(num_embeddings):
        # Compute dot product: embedding . embeddings_matrix[i]
        dot_product = 0.0
        for j in range(embedding_dim):
            dot_product += embedding[j] * embeddings_matrix[i, j]

        # Compute norm of this embedding
        magnitude_B = 0.0
        for j in range(embedding_dim):
            magnitude_B += embeddings_matrix[i, j] * embeddings_matrix[i, j]
        magnitude_B = magnitude_B ** 0.5

        # Compute cosine similarity (embedding_norm is magnitude_A)
        if embedding_norm * magnitude_B == 0.0:
            continue
        cosine_similarity = dot_product / (embedding_norm * magnitude_B)

        # Map to confidence via sigmoid
        confidence = sigmoid(cosine_similarity, median, range_width, 12.0)

        if confidence > best_score:
            best_score = confidence
            best_idx = i

    if best_idx < 0:
        return ("", 0.0)

    return (labels[best_idx], max(0.0, best_score))


def compute_all_cosines(
    double[:] embedding,
    double[:, :] embeddings_matrix,
) -> np.ndarray:
    """Compute cosine similarity between one embedding and a matrix of embeddings.

    Returns a 2D numpy array (N, 2) where each row is [index, cosine_similarity].

    Args:
        embedding: 1D numpy array (embedding_dim,) - the query embedding
        embeddings_matrix: 2D numpy array (num_embeddings, embedding_dim)

    Returns:
        2D numpy array (N, 2) with [index, cosine_similarity] rows
    """
    cdef:
        int embedding_dim = embedding.shape[0]
        int n = embeddings_matrix.shape[0]
        double embedding_norm = 0.0
        double[:, :] result_view
        int i, j
        double dot_product
        double magnitude_B

    if n == 0:
        return np.empty((0, 2), dtype=np.float64)

    result = np.zeros((n, 2), dtype=np.float64)
    result_view = result

    # Compute embedding norm once
    for j in range(embedding_dim):
        embedding_norm += embedding[j] * embedding[j]
    embedding_norm = embedding_norm ** 0.5

    for i in range(n):
        # Dot product
        dot_product = 0.0
        for j in range(embedding_dim):
            dot_product += embedding[j] * embeddings_matrix[i, j]

        # Norm of embedding row
        magnitude_B = 0.0
        for j in range(embedding_dim):
            magnitude_B += embeddings_matrix[i, j] * embeddings_matrix[i, j]
        magnitude_B = magnitude_B ** 0.5

        if embedding_norm * magnitude_B == 0.0:
            continue

        cosine = dot_product / (embedding_norm * magnitude_B)
        result_view[i, 0] = i
        result_view[i, 1] = cosine

    return result
