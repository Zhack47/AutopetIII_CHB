
import numpy as np
import json
from enum import Enum
from scipy.ndimage import median_filter
import SimpleITK as sitk

class Tracer(Enum):
    FDG = 1,
    PSMA = 2,
    UKN = 3
     
class PETProfiler:
    def __init__(self, background_threshold) -> None:
        self.background_threshold = background_threshold
    
    def __call__(self, arr):
        # Maximum Intensity Projection with NaN bg
        mip_arr = np.max(arr, axis=1)
        nan_mip = np.where(mip_arr<self.background_threshold, np.nan, mip_arr)
        
        # arms removal
        isnan_map = np.isnan(nan_mip)
        ccmap = np.zeros_like(nan_mip)
        
        # for each volume slice on the vertical axis 
        for i in range(isnan_map.shape[0]):
            class_index = 0
            last_index = False
            # we count connected components
            for j in range(isnan_map.shape[1]):
                if not isnan_map[i][j]:
                    if not last_index:
                        class_index += 1
                    ccmap[i][j] = class_index
                last_index = not isnan_map[i][j]
            
            # we only keep the cc closest to the middle of the slice (slice is 1D cause we are working on a mip)
            mid_class_index = ccmap[i][isnan_map.shape[1]//2] if ccmap[i][isnan_map.shape[1]//2] > 0 else 1
            ccmap[i] = np.where(ccmap[i] == mid_class_index, 1,np.nan)    
        nan_mip = ccmap * nan_mip
             
        return np.nanmedian(nan_mip, axis = 1), mip_arr, ccmap
        

class StatisticalDiscriminator:
    def __init__(self, params) -> None:
        self.background_threshold = params["bg_thresh"]
        self.fdg_mean = params["fdg"]["mean"]
        self.psma_mean = params["psma"]["mean"]
    
    def __call__(self, arr):
        nan_arr = np.where(arr<self.background_threshold, np.nan, arr)
        arr_mean = np.nanmean(nan_arr)
        if abs(self.psma_mean - arr_mean) < abs(self.fdg_mean - arr_mean):
            return Tracer.PSMA
        return Tracer.FDG
        
class SpatialDiscriminator:
    def __init__(self, params) -> None:
        self.fdg_spacings = params["fdg"]
        self.psma_spacings = params["psma"]
    
    def __call__(self, spacing):
        spacing = np.asarray(spacing)
        fdg_dist = []
        for fdg_s in self.fdg_spacings:
            fdg_dist = np.linalg.norm(spacing - fdg_s)
        fdg_min = np.min(fdg_dist)
        
        psma_dist = []
        for psma_s in self.psma_spacings:
            psma_dist = np.linalg.norm(spacing - psma_s)
        psma_min = np.min(psma_dist)
        
        if psma_min < fdg_min :
            return Tracer.PSMA
        return Tracer.FDG

class ProfileDiscriminator:
    def __init__(self, params) -> None:
        self.profiler = PETProfiler(background_threshold=params["bg_thresh"])
        self.ratio_threshold = params["ratio_thresh"]
    
    def __call__(self, arr):
        
        profile, _, _  = self.profiler(arr=arr)
        # keep only the modals in the profile
        profile = np.nan_to_num(profile)
        profile = profile - np.mean(profile) 
        modals = np.where(profile <0, 0, 1)
        
        # median filter on the modals to
        modals = median_filter(modals, size=20)
        
        # count the modals
        lastx = 0
        modal_index = 0
        for slice_index in range(len(modals)):
            if modals[slice_index] == 1:
                if lastx == 0:
                    modal_index +=1
                modals[slice_index] = modal_index
            lastx = modals[slice_index]
            
        # register modal maxima
        modal_maxima = []
        for i in np.unique(modals)[1:]:
            modal_maxima.append(np.max(profile[np.where(modals==i)]))
        
        # handle cases without exactly two modals
        if len(modal_maxima) == 1:
            #modal_maxima.insert(0,0) # most 1 modals are FDG
            return Tracer.UKN
        if len(modal_maxima) > 2:
            mmidx = sorted([np.argpartition(modal_maxima, kth = -2)[-2], np.argpartition(modal_maxima, kth = -1)[-1]])
            modal_maxima = [modal_maxima[mmidx[0]], modal_maxima[mmidx[1]]]
        
        # compute modal ratio
        if modal_maxima[0] == 0:
            modal_maxima[0] = 1e-3
        modal_ratio = modal_maxima[1]/modal_maxima[0]
        #print(modal_ratio, self.ratio_threshold)
        if modal_ratio > self.ratio_threshold:
            return Tracer.FDG
        return Tracer.PSMA
    
class TracerDiscriminator:
    
    def __init__(self, params_path) -> None:
        self.load_params(params_path)
        self.profileD = ProfileDiscriminator(self.params["profile"])
        self.spatialD = SpatialDiscriminator(self.params["spatial"])
        self.statisticalD = StatisticalDiscriminator(self.params["statistical"])

    def __call__(self, arr, spacing):
        profile_decision = self.profileD(arr)
        spatial_decision = self.spatialD(spacing)
        statistical_decision = self.statisticalD(arr)
        
        decisions = [
            (profile_decision, self.params["weights"]["profile"]),
            (spatial_decision, self.params["weights"]["spatial"]),
            (statistical_decision, self.params["weights"]["statistical"])
            ]
        
        fdg_vote = .0
        psma_vote = .0
        
        for d,w  in decisions:
            fdg_vote +=  int(d==Tracer.FDG)*w 
            psma_vote +=  int(d==Tracer.PSMA)*w 
        
        if psma_vote > fdg_vote:
            final_decision = Tracer.PSMA
        else:
            final_decision = Tracer.FDG
        
        split = {
            "profile": profile_decision,
            "spatial": spatial_decision,
            "statistical": statistical_decision
        }
        return final_decision, split

    
    def load_params(self, path):
        with open(path) as f:
            self.params = json.load(f)
        
    
    