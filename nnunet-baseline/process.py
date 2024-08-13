import json
import os
import shutil
import subprocess

import SimpleITK
import numpy as np
import torch
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import os
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, subfiles, join


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

    def predict(self):
        """
        Your algorithm goes here
        """
        print("nnUNet segmentation starting!")

        os.environ['nnUNet_compile'] = 'F'  # on my system the T does the test image in 2m56 and F in 3m15. Not sure if
        # 20s is worth the risk

        maybe_mkdir_p(self.output_path)

        trained_model_path = "nnUNet_results/Dataset512_AUTOPETIII_SUPLAB_WIN/nnUNetTrainer__nnUNetPlans__3d_fullres"

        ct_mha = subfiles(join(self.input_path, 'images/ct/'), suffix='.mha')[0]
        pet_mha = subfiles(join(self.input_path, 'images/pet/'), suffix='.mha')[0]
        uuid = os.path.basename(os.path.splitext(ct_mha)[0])
        output_file_trunc = os.path.join(self.output_path + uuid)

        predictor = nnUNetPredictor(
            tile_step_size=0.6,
            use_mirroring=False,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True)
        predictor.initialize_from_trained_model_folder(trained_model_path, use_folds=(0,1,2,3,4))
        predictor.dataset_json['file_ending'] = '.mha'

        # ideally we would like to use predictor.predict_from_files but this stupid docker container will be called
        # for each individual test case so that this doesn't make sense
        images, properties = SimpleITKIO().read_images([ct_mha, pet_mha])
        ct = images[0]
        pt = images[1]
        ct_win = np.clip(ct, -300, 400)
        pt_win = np.clip(pt, 0, 20)
        images = np.stack([ct, pt, ct_win, pt_win])
        predictor.predict_single_npy_array(images, properties, None, output_file_trunc, False)

        # Keeping only the 'lesion' class
        out_image = SimpleITK.ReadImage(output_file_trunc+".mha")
        out_np = SimpleITK.GetArrayFromImage(out_image)
        oneclass_np = np.zeros_like(out_np)
        oneclass_np[out_np==1] = 1
        oneclass_image = SimpleITK.GetImageFromArray(oneclass_np)
        oneclass_image.CopyInformation(out_image)
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
