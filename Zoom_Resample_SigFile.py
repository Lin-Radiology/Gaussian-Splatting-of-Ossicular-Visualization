import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, uniform_filter
import os

# =============================
# 路径
# =============================
img_path  = "/Users/id/Desktop/Inner_Ear_Pathology/0042606917_0000.nii.gz"
mask_path = "/Users/id/Desktop/Inner_Ear_Pathology/0042606917/0042606917_L.nii.gz"

out_dir = "/Users/id/Desktop/Inner_Ear_Pathology/0042606917"
os.makedirs(out_dir, exist_ok=True)

out_img_path  = os.path.join(out_dir, "0042606917_0000_resample_L.nii.gz")
out_mask_path = os.path.join(out_dir, "0042606917_0000_resample_mask_L.nii.gz")

print("Loading image and mask...")
img  = nib.load(img_path)
mask = nib.load(mask_path)

# =============================
# 统一到 RAS
# =============================
img  = nib.as_closest_canonical(img)
mask = nib.as_closest_canonical(mask)

data = img.get_fdata(dtype=np.float32)
mask_data = mask.get_fdata() > 0

affine = img.affine.copy()
shape = data.shape

print("Image shape (X, Y, Z):", shape)
print("Original voxel sizes:", img.header.get_zooms())

# =============================
# Mask 边界扩展 + soft mask
# =============================
mask_dilated = binary_dilation(mask_data, iterations=2)

soft_mask = uniform_filter(mask_dilated.astype(np.float32), size=5)
soft_mask = np.clip(soft_mask, 0, 1)

data_masked = data * soft_mask

# =============================
# 基于 dilated mask 做真实空间裁剪
# =============================
x_any = mask_dilated.any(axis=(1, 2))
y_any = mask_dilated.any(axis=(0, 2))
z_any = mask_dilated.any(axis=(0, 1))

x_idx = np.where(x_any)[0]
y_idx = np.where(y_any)[0]
z_idx = np.where(z_any)[0]

if len(x_idx) == 0 or len(y_idx) == 0 or len(z_idx) == 0:
    raise ValueError("Mask is empty! Cannot crop.")

xmin, xmax = x_idx[0], x_idx[-1]
ymin, ymax = y_idx[0], y_idx[-1]
zmin, zmax = z_idx[0], z_idx[-1]

margin = 8
x0, x1 = max(0, xmin - margin), min(shape[0], xmax + margin + 1)
y0, y1 = max(0, ymin - margin), min(shape[1], ymax + margin + 1)
z0, z1 = max(0, zmin - margin), min(shape[2], zmax + margin + 1)

# =============================
# 同步裁剪 Image 和 Mask
# =============================
cropped_img  = data_masked[x0:x1, y0:y1, z0:z1]
cropped_mask = mask_data[x0:x1, y0:y1, z0:z1].astype(np.uint8)

print("Cropped image shape:", cropped_img.shape)
print("Cropped mask shape :", cropped_mask.shape)

# =============================
# 修正 affine（只改平移，不改比例）
# =============================
new_affine = affine.copy()
translation_vector = np.array([x0, y0, z0])
new_affine[:3, 3] = affine[:3, 3] + affine[:3, :3] @ translation_vector

# =============================
# 保存 Image（header 重建）
# =============================
out_img = nib.Nifti1Image(cropped_img.astype(np.float32), new_affine)
out_img.set_sform(new_affine, code=1)
out_img.set_qform(new_affine, code=1)
nib.save(out_img, out_img_path)

# =============================
# 保存 Mask（严格 0/1，无 soft）
# =============================
out_mask = nib.Nifti1Image(cropped_mask, new_affine)
out_mask.set_sform(new_affine, code=1)
out_mask.set_qform(new_affine, code=1)
nib.save(out_mask, out_mask_path)

print("✔ Complete")
print("✔ Image output :", out_img_path)
print("✔ Mask  output :", out_mask_path)