import numpy as np
import nibabel as nib
import SimpleITK as sitk

pet_sitk = sitk.ReadImage("/home/zhack/Documents/THESE/rtep7^005-002_2019-10-09_(SUVbw).nii.gz")
pet_np = sitk.GetArrayFromImage(pet_sitk)

def threshold_bounding_box(arr, threshold=.1):
    threshold_indices = np.where(arr > threshold)
    min_indices = [np.min(indices) for indices in threshold_indices]
    max_indices = [np.max(indices) for indices in threshold_indices]
    return tuple(min_indices), tuple(max_indices)

(x_min, y_min, z_min), (x_max, y_max, z_max) = threshold_bounding_box(pet_np, .1)
pet_cut = pet_np[x_min: x_max, y_min:y_max, z_min:z_max]
seg = (pet_cut>2.5)*1
out_seg =np.zeros_like(pet_np)
out_seg[x_min:x_max, y_min:y_max, z_min:z_max] = seg
print(pet_np.shape)
print(out_seg.shape)
out_seg_path = "/home/zhack/Documents/THESE/pet_seg-005-003.nii.gz"
inter_seg_path = "/home/zhack/Documents/THESE/pet_seg-005-003_inter.nii.gz"
image_out = sitk.GetImageFromArray(out_seg.astype(np.uint8))
image_out.CopyInformation(pet_sitk)

inter_image = sitk.GetImageFromArray(pet_cut)

src_spacing = pet_sitk.GetSpacing()
src_origin = pet_sitk.GetOrigin()
src_direction = pet_sitk.GetDirection()

x_mod = src_origin[0]+src_spacing[0]*src_direction[0] * z_min  # This is
y_mod = src_origin[1]+src_spacing[1]*src_direction[4] * y_min  # SimpleITK's
z_mod = src_origin[2]+src_spacing[2]*src_direction[8] * x_min  # fault
dst_origin = (x_mod, y_mod, z_mod)

inter_image.SetSpacing(pet_sitk.GetSpacing())
inter_image.SetOrigin(dst_origin)
inter_image.SetDirection(pet_sitk.GetDirection())

sitk.WriteImage(image_out, out_seg_path)
sitk.WriteImage(inter_image, inter_seg_path)
