from deep_discriminator import DeepDiscriminator
import numpy as np
import SimpleITK as sitk
import torch
import os
from enum import Enum

class Tracer(Enum):
    FDG = 0
    PSMA = 1

def resample_img(itk_image, out_spacing=[3.0, 3.0, 3.0], is_label=False):
    
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)


class SmartTracerDiscriminator:
    def __init__(self, weights_path, device, target_spacing = [3.0, 3.0, 3.0]):
        self.model_weights = []
        for filename in os.listdir(weights_path):
            self.model_weights.append(os.path.join(weights_path, filename))
        
        self.target_spacing = target_spacing
        self.img_size = [224,224]
        
        self.device = device
        self.model = DeepDiscriminator().to(device)
        self.model.eval()
    
    def __call__(self, sitk_img):
        # Preprocessing
        res_img = resample_img(sitk_img, out_spacing=self.target_spacing)
        vol_arr = sitk.GetArrayFromImage(res_img)
        mip_arr= np.max(vol_arr, axis=1)
        
        center = np.asarray(mip_arr.shape)//2
        og = center - np.asarray(self.img_size)//2
        og  = np.where(og < 0, 0, og)      
        windowed_arr =  mip_arr[og[0]:og[0]+self.img_size[0], og[1]:og[1]+self.img_size[1]]
        
        x = np.zeros(tuple(self.img_size))
        x[0:windowed_arr.shape[0],0:windowed_arr.shape[1]] = windowed_arr
        x = torch.tensor(x, dtype=torch.float)[None,None,...].to(self.device)
        
        # Prediction
        preds = []
        for fold_weights in self.model_weights:
            # Load weights
            self.model.load_state_dict(torch.load(fold_weights, weights_only=True))
            
            # Prediction
            with torch.no_grad():
                output = self.model(x)
                
            # Postpocessing
            preds.append(torch.round(output).int().item())

        return Tracer(np.argmax(np.bincount(preds)))
        