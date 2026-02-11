import numpy as np
import os

DISTANCE_NPY_LIST = [
    "/Users/id/Desktop/软件毕设/0042599621_L_NBK/geometry_eval/surface_to_gaussian_distances.npy",
    "/Users/id/Desktop/软件毕设/0013143741_R_NBK/geometry_eval/surface_to_gaussian_distances.npy",
    "/Users/id/Desktop/软件毕设/0000264915_R_NBK/geometry_eval/surface_to_gaussian_distances.npy"
]

THRESHOLD_MM = 0.3

coverages = []

print("===== Surface Coverage Evaluation =====")

for i, npy_path in enumerate(DISTANCE_NPY_LIST, start=1):
    distances = np.load(npy_path)
    coverage = np.mean(distances < THRESHOLD_MM)
    coverages.append(coverage)

    print(f"Case {i}: Coverage @ {THRESHOLD_MM} mm = {coverage*100:.2f}%")

coverages = np.array(coverages)

print("\nOverall:")
print(
    f"Coverage @ {THRESHOLD_MM} mm: "
    f"{np.mean(coverages)*100:.2f}% ± {np.std(coverages)*100:.2f}%"
)