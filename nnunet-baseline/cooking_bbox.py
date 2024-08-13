import numpy as np
import nibabel as nib
import SimpleITK as sitk

pet_sitk = sitk.ReadImage("/home/zhack/Documents/THESE/rtep7^005-002_2019-10-09_(SUVbw).nii.gz")
pet_np = sitk.GetArrayFromImage(pet_sitk)

def threshold_bounding_box(arr, threshold=.1):
    threshold_indices = np.where(arr > threshold)
    print(threshold_indices)
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

inter_image.SetSpacing(pet_sitk.GetSpacing())
inter_image.SetOrigin(pet_sitk.GetOrigin())
inter_image.SetDirection(pet_sitk.GetDirection())
print(pet_sitk.GetPixelIDValue())
print(inter_image.GetPixelIDValue())
print(image_out.GetPixelIDValue())
sitk.WriteImage(image_out, out_seg_path)
sitk.WriteImage(inter_image, inter_seg_path)
