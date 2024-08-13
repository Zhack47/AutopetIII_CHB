import numpy as np
import nibabel as nib
import SimpleITK as sitk

pet_reader = sitk.ImageSeriesReader()
fnames = pet_reader.GetGDCMSeriesFileNames("/media/zhack/DD1/rtep7_newhope/005-002/PT/AQUILAB/1.3.46.670589.28.2.14.18447.53045.64458.2.368.0.1570692895")
pet_reader.SetFileNames(fnames)
pet_sitk = pet_reader.Execute()
pet_np = sitk.GetArrayFromImage(pet_sitk)
def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    dpt = np.any(img, axis=2)
    print(img.shape)
    print(rows.shape)
    print(cols.shape)
    print(dpt.shape)
    exit()
    zmin, zmax = np.where(dpt)[0][[0, -1]]
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return img[ymin:ymax+1, xmin:xmax+1, zmin:zmax+1]

bbox2(pet_np)