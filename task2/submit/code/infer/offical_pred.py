import copy
from pathlib import Path
import SimpleITK as sitk
import numpy as np
import torch
import torch.nn as nn
import json
import glob
import os
import argparse

from typing import Dict, List, Tuple, Union

from evalutils import SegmentationAlgorithm
from acvl_utils.morphology.morphology_helper import remove_all_but_largest_component, remove_components
from scripts.infer_iac import InferClass

def get_default_device():
    """Set device for computation"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def region_or_label_to_mask(segmentation: np.ndarray, region_or_label: Union[int, Tuple[int, ...]]) -> np.ndarray:
    if np.isscalar(region_or_label):
        return segmentation == region_or_label
    else:
        mask = np.zeros_like(segmentation, dtype=bool)
        for r in region_or_label:
            mask[segmentation == r] = True
    return mask

def remove_all_but_largest_component_from_segmentation(segmentation: np.ndarray,
                                                       labels_or_regions: Union[int, Tuple[int, ...],
                                                                                List[Union[int, Tuple[int, ...]]]],
                                                       background_label: int = 0) -> np.ndarray:    
    mask = np.zeros_like(segmentation, dtype=bool)
    if not isinstance(labels_or_regions, list):
        labels_or_regions = [labels_or_regions]
    for l_or_r in labels_or_regions:
        mask |= region_or_label_to_mask(segmentation, l_or_r)
    mask_keep = remove_all_but_largest_component(mask)
    ret = np.copy(segmentation)  # do not modify the input!
    ret[mask & ~mask_keep] = background_label
    return ret

def remove_components_from_segmentation(segmentation: np.ndarray,
                                    labels_or_regions: Union[int, Tuple[int, ...],
                                                            List[Union[int, Tuple[int, ...]]]],
                                    background_label: int = 0) -> np.ndarray:    
    mask = np.zeros_like(segmentation, dtype=bool)
    if not isinstance(labels_or_regions, list):
        labels_or_regions = [labels_or_regions]
    for l_or_r in labels_or_regions:
        mask |= region_or_label_to_mask(segmentation, l_or_r)
    mask_keep = remove_components(mask,1000)
    ret = np.copy(segmentation)  # do not modify the input!
    ret[mask & ~mask_keep] = background_label
    return ret

class ToothFairy3_OralPharyngealSegmentation(SegmentationAlgorithm):
    def __init__(self, debug=False):
        if debug:
            print("DEBUG MODE")
            super().__init__(
                input_path=Path('./test/input/images/cbct/'),
                output_path=Path('./output/'),
                output_file=Path('./output/result.json'),
                validators={},
            )
        else:
            super().__init__(
                input_path=Path('/input/images/cbct/'),
                output_path=Path('/output/images/iac-segmentation/'),
                validators={},
            )

        # Create output directory if it doesn't exist
        if not self._output_path.exists():
            self._output_path.mkdir(parents=True)

        # Create metadata output directory
        self.metadata_output_path = Path('/output/metadata/') if not debug else Path('./output/metadata/')
        if not self.metadata_output_path.exists():
            self.metadata_output_path.mkdir(parents=True)

        # Initialize device
        self.device = get_default_device()
        print(f"Using device: {self.device}")
        
        self.debug = debug
        self.infer = InferClass() if not debug else InferClass(config_file='./configs/infer_iac_debug.yaml')

    def save_instance_metadata(self, metadata: Dict, image_name: str):
        """
        Save instance metadata to JSON file

        Args:
            metadata: Instance metadata dictionary
            image_name: Name of the input image (without extension)
        """
        metadata_file = self.metadata_output_path / f"{image_name}_instances.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def process_case(self, *, idx, case):
        # Load and test the image for this case
        input_image, input_image_file_path = self._load_input_image(case=case)

        # Segment nodule candidates
        segmented_ = self.predict(input_image=input_image, input_image_file_path=input_image_file_path)

        # Write resulting segmentation to output location
        segmentation_path = self._output_path / input_image_file_path.name
        if not self._output_path.exists():
            self._output_path.mkdir()
        sitk.WriteImage(segmented_, str(segmentation_path), True)

        # Write segmentation file path to result.json for this case
        return {
            "outputs": [
                dict(type="metaio_image", filename=segmentation_path.name)
            ],
            "inputs": [
                dict(type="metaio_image", filename=input_image_file_path.name)
            ],
            "error_messages": [],
        }

    def convert_clicks(self, point_json, label="Left_IAC"):
        label_num = 133 if label == "Left_IAC" else 134
        clicks = [{"fg": [point['point'] for point in point_json['points'] if point['name'] == 'Left_IAC'], 
            'bg': [point['point'] for point in point_json['points'] if point['name'] != 'Left_IAC']}]
        data = clicks
        B = len(data)  # Number of objects
        indexes = np.arange(1, B + 1).tolist()
        # Determine the maximum number of points across all objects
        max_N = max(len(obj["fg"]) + len(obj["bg"]) for obj in data)

        # Initialize padded arrays
        point_coords = np.zeros((B, max_N, 3), dtype=int)
        point_labels = np.full((B, max_N), -1, dtype=int)
        
        for i, obj in enumerate(data):
            points = []
            labels = []

            # Add foreground points
            for fg_point in obj["fg"]:
                fg_point = [fg_point[2], fg_point[1], fg_point[0]]
                points.append(fg_point)
                labels.append(133)

            # Add background points
            for bg_point in obj["bg"]:
                bg_point = [bg_point[2], bg_point[1], bg_point[0]]
                
                points.append(bg_point)
                labels.append(134)

            if len(points) > 0:
                point_coords[i, : len(points)] = points
                point_labels[i, : len(labels)] = labels        
        
        label_prompt = np.array([label_num])[np.newaxis, ...]
        prompt_class = copy.deepcopy(label_prompt)
        # label_prompt = np.array([1,2])
        # prompt_class = None
        return point_coords, point_labels, indexes, label_prompt, prompt_class


    @torch.no_grad()
    def predict(self, *, input_image: sitk.Image, input_image_file_path: str = None) -> sitk.Image:
        # === Load and parse the JSON clicks file ===

        filename = Path(input_image_file_path).name

        if filename.endswith(".nii.gz"):
            base = filename[:-7]  # remove '.nii.gz' (7 chars)
        elif filename.endswith(".mha"):
            base = filename[:-4]  # remove '.mha' (4 chars)
        else:
            raise ValueError("Unsupported file extension")

        point_json_root = '/input/' if not self.debug else './test/input/'
        parts = base.split('_')
        input_json_clicks = f"{point_json_root}iac_clicks_{parts[0]}_{parts[-1]}.json"
        if not os.path.isfile(input_json_clicks):
            input_json_clicks = f"{point_json_root}iac_clicks_{base}.json"
        if not os.path.isfile(input_json_clicks):
            # Look for exactly one JSON file in /input/ that has the keyword "clicks"
            json_files = [f for f in glob.glob(point_json_root+"*.json") if "clicks" in f]
            print(json_files)
            if len(json_files) == 1:
                input_json_clicks = json_files[0]
                print(f"Using single JSON file found: {input_json_clicks}")
            else:
                raise RuntimeError(f"Could not find clicks JSON file at '{input_json_clicks}', "
                                   f"and found {len(json_files)} JSON files in /input/: {json_files}")

        try:
            with open(input_json_clicks, 'r') as f:
                clicks_data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load JSON clicks file '{input_json_clicks}': {e}")

        if str(input_image_file_path).endswith('mha'):
            sitk.WriteImage(input_image, 'image.nii.gz')
            input_image_file_path = 'image.nii.gz'
        if len(clicks_data['points']) == 0:
            output_array = self.infer.infer_everything(input_image_file_path, label_prompt=[133,134],save_mask=False)
            output_array = output_array[0].transpose(2, 0)
            self.infer.batch_data = None
            self.infer.prev_mask = None
        else:
            self.infer.infer_everything(input_image_file_path, label_prompt=[133],save_mask=False)
            point_coords, point_labels, indexes, label_prompt, prompt_class = self.convert_clicks(clicks_data)      
            output_array_left = self.infer.infer(input_image_file_path, point=point_coords, point_label=point_labels, label_prompt=label_prompt, prompt_class=prompt_class, save_mask=False)
            output_array_left = output_array_left[0].transpose(2, 0)
            self.infer.batch_data = None
            self.infer.prev_mask = None
            self.infer.infer_everything(input_image_file_path, label_prompt=[134],save_mask=False)
            point_coords, point_labels, indexes, label_prompt, prompt_class = self.convert_clicks(clicks_data,label='Right_IAC')
            output_array_right = self.infer.infer(input_image_file_path, point=point_coords, point_label=point_labels, label_prompt=label_prompt, prompt_class=prompt_class, save_mask=False)
            output_array_right = output_array_right[0].transpose(2, 0)
            self.infer.batch_data = None
            self.infer.prev_mask = None
            output_array = output_array_left + output_array_right

        output_array = torch.clamp(output_array.int(), 0, 255).numpy().astype(np.uint8) 
        output_array = remove_all_but_largest_component_from_segmentation(output_array, [133])
        output_array = remove_all_but_largest_component_from_segmentation(output_array, [134])
        output_array[output_array == 133] = 1
        output_array[output_array == 134] = 2
        output_array[output_array > 2] = 0
        output_array[output_array < 0] = 0
        output_image = sitk.GetImageFromArray(output_array)
        output_image.CopyInformation(input_image)
        return output_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=bool, default=False)
    args = parser.parse_args()
    ToothFairy3_OralPharyngealSegmentation(debug=args.debug).process()