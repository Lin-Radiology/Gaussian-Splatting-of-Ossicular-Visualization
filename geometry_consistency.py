import os
import numpy as np
import nibabel as nib
from skimage import measure
import trimesh
import open3d as o3d
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt


# =========================
# 配置路径（3例）
# =========================
MASK_PATHS = [
    "/Users/id/Desktop/软件毕设/0042599621_masked_aligned_L.nii.gz",
    "/Users/id/Desktop/软件毕设/0013143741_masked_aligned_R.nii.gz",
    "/Users/id/Desktop/软件毕设/0000264915_masked_aligned_R.nii.gz"
]

GAUSSIAN_PLYS = [
    "/Users/id/Desktop/软件毕设/0042599621_L_NBK/point_cloud/iteration_30000/point_cloud.ply",
    "/Users/id/Desktop/软件毕设/0013143741_R_NBK/point_cloud/iteration_30000/point_cloud.ply",
    "/Users/id/Desktop/软件毕设/0000264915_R_NBK/point_cloud/iteration_30000/point_cloud.ply"
]

OUTPUT_DIRS = [
    "/Users/id/Desktop/软件毕设/0042599621_L_NBK/geometry_eval",
    "/Users/id/Desktop/软件毕设/0013143741_R_NBK/geometry_eval",
    "/Users/id/Desktop/软件毕设/0000264915_R_NBK/geometry_eval"
]

N_SURFACE_POINTS = 20000


# =========================
# 主循环
# =========================
for idx, (mask_path, ply_path, out_dir) in enumerate(
    zip(MASK_PATHS, GAUSSIAN_PLYS, OUTPUT_DIRS)
):
    print("\n" + "=" * 60)
    print(f"[CASE {idx+1}] Processing:")
    print(mask_path)
    print("=" * 60)

    os.makedirs(out_dir, exist_ok=True)

    # =========================
    # Step 1: mask -> surface mesh (CT physical space, mm)
    # =========================
    nii = nib.load(mask_path)
    mask = nii.get_fdata() > 0
    spacing = nii.header.get_zooms()

    verts, faces, _, _ = measure.marching_cubes(
        mask.astype(np.uint8),
        level=0.5,
        spacing=spacing
    )

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.export(os.path.join(out_dir, "segmentation_surface.ply"))

    print(f"[INFO] Surface mesh vertices: {len(verts)}")

    # =========================
    # Step 2: load Gaussian point cloud
    # =========================
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)

    print(f"[INFO] Gaussian points (raw): {points.shape[0]}")

    # =========================
    # Step 3: Gaussian → CT space alignment (center + scale)
    # =========================
    mesh_center = mesh.vertices.mean(axis=0)
    points_center = points.mean(axis=0)

    mesh_centered = mesh.vertices - mesh_center
    points_centered = points - points_center

    mesh_scale = np.linalg.norm(mesh_centered, axis=1).mean()
    points_scale = np.linalg.norm(points_centered, axis=1).mean()

    scale_factor = mesh_scale / points_scale
    points_aligned = points_centered * scale_factor + mesh_center

    print(f"[INFO] Applied scale factor: {scale_factor:.4f}")

    # 保存对齐后的点云（可视化 & sanity check）
    pcd_aligned = o3d.geometry.PointCloud()
    pcd_aligned.points = o3d.utility.Vector3dVector(points_aligned)
    o3d.io.write_point_cloud(
        os.path.join(out_dir, "gaussian_points_aligned.ply"),
        pcd_aligned
    )

    # =========================
    # Step 4: surface sampling
    # =========================
    surface_points, _ = trimesh.sample.sample_surface(
        mesh, N_SURFACE_POINTS
    )

    # =========================
    # Step 5: distance computation
    # =========================
    tree = cKDTree(points_aligned)
    distances, _ = tree.query(surface_points, k=1)

    np.save(
        os.path.join(out_dir, "surface_to_gaussian_distances.npy"),
        distances
    )

    # =========================
    # Step 6: statistics
    # =========================
    mean_d = np.mean(distances)
    median_d = np.median(distances)
    p90_d = np.percentile(distances, 90)

    print("----- Geometric Consistency Statistics -----")
    print(f"Mean distance    : {mean_d:.3f} mm")
    print(f"Median distance  : {median_d:.3f} mm")
    print(f"90th percentile  : {p90_d:.3f} mm")

    # 保存统计到 txt（方便论文汇总）
    with open(os.path.join(out_dir, "geometry_stats.txt"), "w") as f:
        f.write(f"Mean distance (mm): {mean_d:.4f}\n")
        f.write(f"Median distance (mm): {median_d:.4f}\n")
        f.write(f"90th percentile (mm): {p90_d:.4f}\n")

    # =========================
    # Step 7: Histogram
    # =========================
    plt.figure()
    plt.hist(distances, bins=100, density=True)
    plt.xlabel("Surface-to-Gaussian Distance (mm)")
    plt.ylabel("Probability Density")
    plt.title("Geometric Consistency Histogram")
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "distance_histogram.png"),
        dpi=300
    )
    plt.close()

    # =========================
    # Step 8: CDF curve
    # =========================
    sorted_d = np.sort(distances)
    cdf = np.arange(len(sorted_d)) / len(sorted_d)

    plt.figure()
    plt.plot(sorted_d, cdf)
    plt.xlabel("Surface-to-Gaussian Distance (mm)")
    plt.ylabel("Cumulative Fraction")
    plt.title("Geometric Consistency CDF")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "distance_cdf.png"),
        dpi=300
    )
    plt.close()

    print(f"[INFO] Case {idx+1} finished.")

    # ==========================================================
    # Step 9: Aggregate results & plot combined CDF (SCI-ready)
    # ==========================================================

    ALL_DISTANCES = []
    MEANS = []
    MEDIANS = []
    P90S = []

    CASE_LABELS = ["Case 1", "Case 2", "Case 3"]

    plt.figure(figsize=(6, 4))

    for idx, out_dir in enumerate(OUTPUT_DIRS):
        dist_path = os.path.join(out_dir, "surface_to_gaussian_distances.npy")
        distances = np.load(dist_path)

        ALL_DISTANCES.append(distances)

        mean_d = np.mean(distances)
        median_d = np.median(distances)
        p90_d = np.percentile(distances, 90)

        MEANS.append(mean_d)
        MEDIANS.append(median_d)
        P90S.append(p90_d)

        # CDF
        sorted_d = np.sort(distances)
        cdf = np.arange(len(sorted_d)) / len(sorted_d)

        plt.plot(
            sorted_d,
            cdf,
            linewidth=2,
            label=f"{CASE_LABELS[idx]}"
        )

    # ---- SCI figure styling ----
    plt.xlabel("Surface-to-Gaussian Distance (mm)", fontsize=11)
    plt.ylabel("Cumulative Fraction", fontsize=11)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=False, fontsize=10)
    plt.tight_layout()

    FIG_PATH = "/Users/id/Desktop/软件毕设/Geometric_Consistency_CDF.png"
    plt.savefig(FIG_PATH, dpi=300)
    plt.close()

    print(f"\n[FIGURE SAVED] {FIG_PATH}")

    # ==========================================================
    # Step 10: Console summary for manuscript (Results section)
    # ==========================================================

    print("\n===== Summary of Geometric Consistency (mm) =====")

    for i in range(3):
        print(
            f"{CASE_LABELS[i]}: "
            f"Mean = {MEANS[i]:.3f}, "
            f"Median = {MEDIANS[i]:.3f}, "
            f"90% = {P90S[i]:.3f}"
        )

    print("\nOverall (n = 3):")
    print(
        f"Mean distance: {np.mean(MEANS):.3f} ± {np.std(MEANS):.3f} mm"
    )
    print(
        f"Median distance: {np.mean(MEDIANS):.3f} ± {np.std(MEDIANS):.3f} mm"
    )
    print(
        f"90th percentile: {np.mean(P90S):.3f} ± {np.std(P90S):.3f} mm"
    )

print("\n[ALL DONE] Geometry consistency evaluation completed for all cases.")