import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# =============================================================================
# SVD Image Compression
# =============================================================================
# Any m x n matrix A (our grayscale image) can be factored as:
#
#     A = U @ Sigma @ V.T
#
# where:
#   U     : m x m orthogonal matrix (left singular vectors)
#   Sigma : m x n diagonal matrix (singular values, descending order)
#   V.T   : n x n orthogonal matrix (right singular vectors)
#
# The rank-k approximation keeps only the top k singular values:
#
#     A_k = U_k @ Sigma_k @ V_k.T
#
# where U_k is the first k columns of U, Sigma_k is the top k x k block,
# and V_k.T is the first k rows of V.T.
#
# By the Eckart-Young theorem, A_k is the BEST possible rank-k approximation
# to A in the Frobenius norm sense — no other rank-k matrix is closer to A.
#
# Compression ratio:
#   Original storage : m * n values
#   Compressed storage: k * (m + n + 1) values
#   Ratio = k * (m + n + 1) / (m * n)
# =============================================================================


def load_grayscale_image(filepath):
    """
    Load an image and convert to a normalized grayscale numpy matrix.
    Pixel values are kept as float64 for numerical precision during SVD.
    """
    img = Image.open(filepath).convert('L')  # 'L' mode = grayscale
    return np.array(img, dtype=np.float64)


def compress_image(A, k):
    """
    Compress a grayscale image matrix A using SVD truncated to rank k.

    Parameters:
    -----------
    A : 2D numpy array
        Grayscale image matrix (pixel values)
    k : int
        Number of singular values to retain

    Returns:
    --------
    A_k : 2D numpy array
        Rank-k approximation of A
    singular_values : 1D numpy array
        All singular values of A (descending)
    """
    # Compute full SVD
    # U: (m x m), s: (min(m,n),), Vt: (n x n)
    U, s, Vt = np.linalg.svd(A, full_matrices=True)

    # Truncate to rank k
    U_k = U[:, :k]          # First k columns of U
    s_k = s[:k]             # Top k singular values
    Vt_k = Vt[:k, :]        # First k rows of V.T

    # Reconstruct: A_k = U_k @ diag(s_k) @ Vt_k
    A_k = U_k @ np.diag(s_k) @ Vt_k

    # Clip values to valid pixel range [0, 255]
    A_k = np.clip(A_k, 0, 255)

    return A_k, s


def frobenius_error(A, A_k):
    """
    Compute the relative Frobenius norm error between original and approximation.

    The Frobenius norm of a matrix is the square root of the sum of squared entries.
    Relative error = ||A - A_k||_F / ||A||_F

    This gives a normalized measure of how much information is lost.
    """
    return np.linalg.norm(A - A_k, 'fro') / np.linalg.norm(A, 'fro')


def compression_ratio(m, n, k):
    """
    Compute the compression ratio for a rank-k approximation.

    Original image requires m*n stored values.
    Rank-k approximation requires k*(m + n + 1) stored values:
      - k*m values for U_k
      - k values for the diagonal of Sigma_k
      - k*n values for Vt_k

    A ratio < 1 means we are storing less data than the original.
    """
    original = m * n
    compressed = k * (m + n + 1)
    return compressed / original


def cumulative_energy(singular_values):
    """
    Compute the cumulative energy captured by the top k singular values.

    Energy is defined as the sum of squared singular values.
    This tells us what fraction of the total 'information' in the image
    is captured by keeping k singular values.

    cumulative_energy[k] = sum(s[:k]^2) / sum(s^2)
    """
    s_squared = singular_values ** 2
    total_energy = np.sum(s_squared)
    return np.cumsum(s_squared) / total_energy


# =============================================================================
# Main
# =============================================================================

# Load image
image_path = "Mona_Lisa_GS2.jpg.webp"
A = load_grayscale_image(image_path)
m, n = A.shape
print(f"Image size: {m} x {n} ({m*n:,} total values)")

# Rank values to visualize
k_values = [1, 5, 10, 20, 50, 100, 200]

# Compute singular values once
_, s, _ = np.linalg.svd(A, full_matrices=False)
print(f"Number of singular values: {len(s)}")
print(f"Top 10 singular values: {s[:10].round(1)}")

# =============================================================================
# Figure 1: Reconstructions at different k values
# =============================================================================
fig, axes = plt.subplots(2, 4, figsize=(16, 9))
axes = axes.flatten()

# Original image
axes[0].imshow(A, cmap='gray', vmin=0, vmax=255)
axes[0].set_title(f'Original\n({m}×{n} = {m*n:,} values)', fontsize=10, fontweight='bold')
axes[0].axis('off')

for idx, k in enumerate(k_values):
    A_k, _ = compress_image(A, k)
    error = frobenius_error(A, A_k)
    ratio = compression_ratio(m, n, k)

    axes[idx + 1].imshow(A_k, cmap='gray', vmin=0, vmax=255)
    axes[idx + 1].set_title(
        f'k = {k}\nRatio: {ratio:.3f} | Error: {error:.3f}',
        fontsize=9
    )
    axes[idx + 1].axis('off')

plt.suptitle('SVD Image Compression — Rank-k Approximations', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('reconstructions.png', dpi=200, bbox_inches='tight')
print("Saved: reconstructions.png")

# =============================================================================
# Figure 2: Singular value decay
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot singular values
ax = axes[0]
ax.plot(s, 'b-', linewidth=1.5)
ax.set_xlabel('Index', fontsize=11)
ax.set_ylabel('Singular Value', fontsize=11)
ax.set_title('Singular Value Decay', fontsize=12, fontweight='bold')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
ax.axvline(x=50, color='r', linestyle='--', alpha=0.7, label='k=50')
ax.axvline(x=100, color='g', linestyle='--', alpha=0.7, label='k=100')
ax.legend()

# Plot cumulative energy
ax = axes[1]
energy = cumulative_energy(s)
ax.plot(energy * 100, 'r-', linewidth=1.5)
ax.set_xlabel('Number of Singular Values (k)', fontsize=11)
ax.set_ylabel('Cumulative Energy Captured (%)', fontsize=11)
ax.set_title('Cumulative Energy vs k', fontsize=12, fontweight='bold')
ax.axhline(y=90, color='b', linestyle='--', alpha=0.7, label='90% energy')
ax.axhline(y=95, color='g', linestyle='--', alpha=0.7, label='95% energy')
ax.axhline(y=99, color='r', linestyle='--', alpha=0.7, label='99% energy')
ax.legend()
ax.grid(True, alpha=0.3)

# Annotate where 90%, 95%, 99% energy is reached
for threshold in [0.90, 0.95, 0.99]:
    k_threshold = np.argmax(energy >= threshold) + 1
    print(f"{threshold*100:.0f}% energy captured at k = {k_threshold} "
          f"(compression ratio = {compression_ratio(m, n, k_threshold):.3f})")

plt.tight_layout()
plt.savefig('singular_value_analysis.png', dpi=200, bbox_inches='tight')
print("Saved: singular_value_analysis.png")

# =============================================================================
# Figure 3: Compression ratio vs reconstruction error
# =============================================================================
k_range = list(range(1, min(201, len(s) + 1)))
errors = []
ratios = []

for k in k_range:
    A_k, _ = compress_image(A, k)
    errors.append(frobenius_error(A, A_k))
    ratios.append(compression_ratio(m, n, k))

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(ratios, errors, 'b-', linewidth=2)
ax.set_xlabel('Compression Ratio (< 1 means smaller than original)', fontsize=11)
ax.set_ylabel('Relative Frobenius Error', fontsize=11)
ax.set_title('Compression Ratio vs Reconstruction Error', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axvline(x=1.0, color='r', linestyle='--', alpha=0.7, label='No compression (ratio=1)')
ax.legend()

plt.tight_layout()
plt.savefig('compression_tradeoff.png', dpi=200, bbox_inches='tight')
print("Saved: compression_tradeoff.png")

plt.show()
