import json
import os
import shutil
import subprocess
from trace import Trace

import SimpleITK
import numpy as np
import torch
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor, nnUNetPredictor_efficient
import os
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, subfiles, join
from smart_tracer_discriminator import SmartTracerDiscriminator, Tracer
from scipy.ndimage import label, binary_dilation
from tqdm import tqdm

class Autopet_baseline:

    def __init__(self):
        """
        Write your own input validators here
        Initialize your model etc.
        """
        # set some paths and parameters
        # according to the specified grand-challenge interfaces
        self.input_path = "/input/"
        # according to the specified grand-challenge interfaces
        self.output_path = "/output/images/automated-petct-lesion-segmentation/"
        self.nii_path = (
            "/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/imagesTs"
        )
        self.result_path = (
            "/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/result"
        )
        self.nii_seg_file = "TCIA_001.nii.gz"
        self.output_path_category = "/output/data-centric-model.json"
        pass

    def convert_mha_to_nii(self, mha_input_path, nii_out_path):  # nnUNet specific
        img = SimpleITK.ReadImage(mha_input_path)
        SimpleITK.WriteImage(img, nii_out_path, True)

    def convert_nii_to_mha(self, nii_input_path, mha_out_path):  # nnUNet specific
        img = SimpleITK.ReadImage(nii_input_path)
        SimpleITK.WriteImage(img, mha_out_path, True)

    def check_gpu(self):
        """
        Check if GPU is available
        """
        print("Checking GPU availability")
        is_available = torch.cuda.is_available()
        print("Available: " + str(is_available))
        print(f"Device count: {torch.cuda.device_count()}")
        if is_available:
            print(f"Current device: {torch.cuda.current_device()}")
            print("Device name: " + torch.cuda.get_device_name(0))
            print(
                "Device memory: "
                + str(torch.cuda.get_device_properties(0).total_memory)
            )

    def load_inputs(self):
        """
        Read from /input/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        ct_mha = os.listdir(os.path.join(self.input_path, "images/ct/"))[0]
        pet_mha = os.listdir(os.path.join(self.input_path, "images/pet/"))[0]
        uuid = os.path.splitext(ct_mha)[0]

        self.convert_mha_to_nii(
            os.path.join(self.input_path, "images/ct/", ct_mha),
            os.path.join(self.nii_path, "TCIA_001_0000.nii.gz"),
        )
        self.convert_mha_to_nii(
            os.path.join(self.input_path, "images/pet/", pet_mha),
            os.path.join(self.nii_path, "TCIA_001_0001.nii.gz"),
        )
        return uuid

    def write_outputs(self, uuid):
        """
        Write to /output/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        shutil.move(os.path.join(self.result_path, uuid+".mha"), os.path.join(self.output_path, uuid + ".mha"))
        """self.convert_nii_to_mha(
            os.path.join(self.result_path, self.nii_seg_file),
            os.path.join(self.output_path, uuid + ".mha"),
        )"""
        print("Output written to: " + os.path.join(self.output_path, uuid + ".mha"))

    # POST-PROCESSING
    def get_3D_bb(self, arr: np.ndarray, index, margin):
        h, w, d = arr.shape
        aw = np.argwhere(arr == index)
        x_min = np.min(aw[:, 0])
        y_min = np.min(aw[:, 1])
        z_min = np.min(aw[:, 2])

        x_max = np.max(aw[:, 0])
        y_max = np.max(aw[:, 1])
        z_max = np.max(aw[:, 2])
        return (max(x_min - margin, 0), min(x_max + margin, h),
                max(y_min - margin, 0), min(y_max + margin, w),
                max(z_min - margin, 0), min(z_max + margin, d))

    def suv_40p(self, image: np.ndarray, mask: np.ndarray, prct: float, fixed: int, min_value=0., min_size=None):
        if prct is None and fixed is None:
            raise ValueError(f"Need at least a % threshold or fixed value threshold")
        labeled_volume, num_labels = label(mask)
        for i in tqdm(range(1, num_labels)):
            xmin, xmax, ymin, ymax, zmin, zmax = self.get_3D_bb(labeled_volume, i, 2)
            cut = image[xmin:xmax, ymin:ymax, zmin:zmax]
            cut_mask = mask[xmin:xmax, ymin:ymax, zmin:zmax] == 1
            nb_voxels_init = cut_mask.sum()
            replacing = np.zeros_like(cut)
            if min_size is None:
                min_size= nb_voxels_init
            if nb_voxels_init >= min_size and cut[cut_mask].max() > min_value:
                cut_mask = binary_dilation(binary_dilation(cut_mask))
                if prct is not None:
                    threshold = (prct * max(image[labeled_volume == i]))
                    if fixed is not None:
                        replacing[(cut >= threshold) | (cut > fixed) ] = 1
                    else:
                        replacing[cut >= threshold] = 1
                else:
                    replacing[cut > fixed] = 1
                replacing[cut_mask == 0] = 0
            else:
                pass
            # replacing = find_biggest_connected_component(replacing)
            mask[xmin:xmax, ymin:ymax, zmin:zmax] = replacing
        return mask

    def post_proc_fdg(self, image: np.ndarray, mask: np.ndarray, min_value=0.):
        return self.suv_40p(image, mask, prct=.4, fixed=4, min_value=min_value, min_size=10)

    def post_proc_psma(self, image: np.ndarray, mask: np.ndarray, min_value=0.):
        return self.suv_40p(image, mask, prct=.25, fixed=None, min_size=10, min_value=min_value)

    def post_proc_ukn(self, image: np.ndarray, mask: np.ndarray):
        return mask

    def predict(self):
        """
        Your algorithm goes here
        """

        def threshold_bounding_box(arr, threshold=.1):
            threshold_indices = np.where(arr > threshold)
            min_indices = [np.min(indices) for indices in threshold_indices]
            max_indices = [np.max(indices) for indices in threshold_indices]
            return tuple(min_indices), tuple(max_indices)

        print("nnUNet segmentation starting!")

        os.environ['nnUNet_compile'] = 'F'  # on my system the T does the test image in 2m56 and F in 3m15. Not sure if
        # 20s is worth the risk

        maybe_mkdir_p(self.output_path)

        trained_model_path_psma = "nnUNet_results/Dataset514_AUTOPETIII_SW_PSMA/nnUNetTrainer__nnUNetPlans__3d_fullres"
        trained_model_path_fdg = "nnUNet_results/Dataset513_AUTOPETIII_SW_FDG/nnUNetTrainer_autopetiii__nnUNetPlans__3d_fullres"
        trained_model_path_ukn = "nnUNet_results/Dataset512_AUTOPETIII_SUPLAB_WIN/nnUNetTrainer__nnUNetPlans__3d_fullres"

        ct_mha = subfiles(join(self.input_path, 'images/ct/'), suffix='.mha')[0]
        pet_mha = subfiles(join(self.input_path, 'images/pet/'), suffix='.mha')[0]
        uuid = os.path.basename(os.path.splitext(ct_mha)[0])
        output_file_trunc = os.path.join(self.output_path + uuid)

        print("Creating", end="")
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_mirroring=True,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True)
        """predictor = nnUNetPredictor_efficient(
            tile_step_size=0.6,
            use_mirroring=True,
            verbose=True,
            verbose_preprocessing=True,
            allow_tqdm=True)"""
        print("Done")


        # ideally we would like to use predictor.predict_from_files but this stupid docker container will be called
        # for each individual test case so that this doesn't make sense
        print("Reading images", end="")
        images, properties = SimpleITKIO().read_images([ct_mha, pet_mha])
        print("Done")

        ct = images[0]
        pt = images[1]

        (x_min, y_min, z_min), (x_max, y_max, z_max) = threshold_bounding_box(pt, .1)
        ct = ct[x_min: x_max, y_min:y_max, z_min:z_max]
        pt_cut= pt[x_min: x_max, y_min:y_max, z_min:z_max]
        print(pt.shape)
        print(pt_cut.shape)

        src_spacing = properties["sitk_stuff"]["spacing"]
        src_origin = properties["sitk_stuff"]["origin"]
        src_direction = properties["sitk_stuff"]["direction"]

        # tracer, _ = TracerDiscriminator("params.json")(pt, src_spacing)
        tracer = SmartTracerDiscriminator("dd_weights/weights", torch.device("cuda"))(pt)


        print("[+] Initalizing model")
        print(f"[+] Using model for {tracer}")
        if tracer==Tracer.PSMA:
            target_spacing = tuple(map(float, json.load(open(join(trained_model_path_psma, "plans.json"), "r"))["configurations"][
                "3d_fullres"]["spacing"]))
            predictor.initialize_from_trained_model_folder(trained_model_path_psma, use_folds=(0,1,2,3,4), checkpoint_name="checkpoint_best.pth")
        elif tracer==Tracer.FDG:
            target_spacing = tuple(map(float, json.load(open(join(trained_model_path_fdg, "plans.json"), "r"))["configurations"][
                "3d_fullres"]["spacing"]))
            predictor.initialize_from_trained_model_folder(trained_model_path_fdg, use_folds="all", checkpoint_name="checkpoint_final.pth")
        """elif tracer==Tracer.UKN:
            target_spacing = tuple(map(float, json.load(open(join(trained_model_path_ukn, "plans.json"), "r"))["configurations"][
                "3d_fullres"]["spacing"]))
            predictor.initialize_from_trained_model_folder(trained_model_path_ukn, use_folds=(0,1,2,3,4), checkpoint_name="checkpoint_best.pth")
            predictor.allowed_mirroring_axes = (1, 2)"""

        fin_size = ct.shape
        new_shape = np.array([int(round(i / j * k)) for i, j, k in zip(src_spacing, target_spacing[::-1], fin_size)])
        print(f"Resampled shape: {new_shape}")
        nb_voxels = np.prod(pt_cut.shape)

        #predictor.configuration_manager.configuration["patch_size"] = [32, 32, 32]
        print("Done")
        predictor.dataset_json['file_ending'] = '.mha'
        ## AAAAH mais osef en fait on set la direction plus tard
        x_mod = src_origin[0] + src_spacing[0] * src_direction[0] * z_min  # This is // This is
        y_mod = src_origin[1] + src_spacing[1] * src_direction[4] * y_min  # SimpleITK's  // different in
        z_mod = src_origin[2] + src_spacing[2] * src_direction[8] * x_min  # fault  // nnUNet (;.;)
        dst_origin = (x_mod, y_mod, z_mod)
        properties["sitk_stuff"]["origin"] = dst_origin

        print("Windowing..", end="")
        ct_win = np.clip(ct, -300, 400)
        pt_win = np.clip(pt_cut, 0, 20)
        print("Done")

        print("Stacking..", end="")
        images = np.stack([ct, pt_cut, ct_win, pt_win])
        print("Done")
        if nb_voxels < 4.7e7 or tracer==Tracer.PSMA:
            predictor.predict_single_npy_array(images, properties, None, output_file_trunc, False)
        elif nb_voxels < 6.5e7:
            print("Removing one axis for prediction mirroring")
            predictor.allowed_mirroring_axes = (1, 2)
            predictor.predict_single_npy_array(images, properties, None, output_file_trunc, False)
        else:
            print("Removing all mirroring")
            predictor.allowed_mirroring_axes = None
            predictor.predict_single_npy_array(images, properties, None, output_file_trunc, False)



        out_image = SimpleITK.ReadImage(output_file_trunc+".mha")
        out_np = SimpleITK.GetArrayFromImage(out_image)

        # Get the SUV Mean for liver and spleen
        liver_class = 3
        spleen_class = 6
        if tracer == Tracer.FDG:
            liver_mask = out_np==liver_class
            liver_suv_mean = np.mean(pt_cut[liver_mask])
            print(f"Liver SUVmean is {liver_suv_mean}")
        elif tracer == Tracer.PSMA:
            spleen_mask = out_np==spleen_class
            spleen_suv_mean = np.mean(pt_cut[spleen_mask])
            print(f"Spleen SUVmean is {spleen_suv_mean}")

        # Keeping only the 'lesion' class
        oneclass_np = np.zeros_like(pt)

        oneclass_np[x_min:x_max, y_min:y_max, z_min:z_max] = out_np==1


        if tracer == Tracer.FDG:
            oneclass_np = self.post_proc_fdg(pt, oneclass_np, min_value=min(2.5,liver_suv_mean))
        elif tracer == Tracer.PSMA:
            oneclass_np = self.post_proc_psma(pt, oneclass_np, min_value=min(3,spleen_suv_mean))
        elif tracer == Tracer.UKN:
            oneclass_np = self.post_proc_ukn(pt, oneclass_np)

        oneclass_image = SimpleITK.GetImageFromArray(oneclass_np.astype(np.uint8))
        oneclass_image.SetOrigin(src_origin)
        oneclass_image.SetSpacing(src_spacing)
        oneclass_image.SetDirection(src_direction)

        SimpleITK.WriteImage(oneclass_image, output_file_trunc+".mha")


        print("Prediction finished")

    def save_datacentric(self, value: bool):
        print("Saving datacentric json to " + self.output_path_category)
        with open(self.output_path_category, "w") as json_file:
            json.dump(value, json_file, indent=4)

    def process(self):
        """
        Read inputs from /input, process with your algorithm and write to /output
        """
        # process function will be called once for each test sample
        self.check_gpu()
        print("Start processing")
        uuid = self.load_inputs()
        print("Start prediction")
        self.predict()
        print("Start output writing")
        self.save_datacentric(False)
        #self.write_outputs(uuid)


if __name__ == "__main__":
    print("START")
    Autopet_baseline().process()
