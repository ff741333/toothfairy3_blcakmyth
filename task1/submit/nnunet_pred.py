import argparse
import gc
import os
from pathlib import Path
from queue import Queue
from threading import Thread
import time
from typing import Union, Tuple
from skimage.measure import label, regionprops
from scipy.ndimage import binary_erosion, binary_dilation

import nnunetv2
import numpy as np
import torch
import SimpleITK as sitk
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.utilities.file_and_folder_operations import load_json, join
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian, compute_steps_for_sliding_window
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.postprocessing.remove_connected_components import remove_all_but_largest_component_from_segmentation
from torch._dynamo import OptimizedModule
from torch.backends import cudnn
from tqdm import tqdm


class CustomPredictor(nnUNetPredictor):
    def predict_single_npy_array(self, input_image: np.ndarray, image_properties: dict,
                                 segmentation_previous_stage: np.ndarray = None):
        torch.set_num_threads(7)
        with torch.no_grad():
            self.network = self.network.to(self.device)
            self.network.eval()

            if self.verbose:
                print('preprocessing')
            preprocessor = self.configuration_manager.preprocessor_class(verbose=self.verbose)
            data, _, image_properties = preprocessor.run_case_npy(input_image, None, image_properties,
                                                self.plans_manager,
                                                self.configuration_manager,
                                                self.dataset_json)

            data = torch.from_numpy(data)
            del input_image
            if self.verbose:
                print('predicting')

            predicted_logits = self.predict_preprocessed_image(data)

            if self.verbose: print('Prediction done')

            segmentation = self.convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits,
                                                                                            image_properties)
        return segmentation

    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                             use_folds: Union[Tuple[Union[int, str]], None],
                                             checkpoint_name: str = 'checkpoint_best.pth'):
        """
        This is used when making predictions with a trained model
        """
        if use_folds is None:
            use_folds = nnUNetPredictor.auto_detect_available_folds(model_training_output_dir, checkpoint_name)

        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)

        if isinstance(use_folds, str):
            use_folds = [use_folds]

        parameters = []
        for i, f in enumerate(use_folds):
            f = int(f) if f != 'all' else f
            checkpoint = torch.load(join(model_training_output_dir, f'fold_{f}', checkpoint_name),
                                    map_location=torch.device('cpu'), weights_only=False)
            if i == 0:
                trainer_name = checkpoint['trainer_name']
                configuration_name = checkpoint['init_args']['configuration']
                inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
                    'inference_allowed_mirroring_axes' in checkpoint.keys() else None

            parameters.append(join(model_training_output_dir, f'fold_{f}', checkpoint_name))

        configuration_manager = plans_manager.get_configuration(configuration_name)
        # restore network
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        trainer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                                    trainer_name, 'nnunetv2.training.nnUNetTrainer')
        if trainer_class is None:
            raise RuntimeError(f'Unable to locate trainer class {trainer_name} in nnunetv2.training.nnUNetTrainer. '
                               f'Please place it there (in any .py file)!')
        network = trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=False
        )

        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        if ('nnUNet_compile' in os.environ.keys()) and (os.environ['nnUNet_compile'].lower() in ('true', '1', 't')) \
                and not isinstance(self.network, OptimizedModule):
            print('Using torch.compile')
            self.network = torch.compile(self.network)

    @torch.inference_mode(mode=True)
    def predict_preprocessed_image(self, image):
        empty_cache(self.device)
        data_device = torch.device('cuda:0')
        predicted_logits_device = torch.device('cuda:0')
        gaussian_device = torch.device('cuda:0')
        compute_device = torch.device('cuda:0')

        data, slicer_revert_padding = pad_nd_image(image, self.configuration_manager.patch_size,
                                                   'constant', {'value': 0}, True,
                                                   None)
        del image

        slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

        empty_cache(self.device)

        data = data.to(data_device)
        predicted_logits = torch.zeros((self.label_manager.num_segmentation_heads-31, *data.shape[1:]),
                                       dtype=torch.int8,
                                       device=predicted_logits_device)
        gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
                                    value_scaling_factor=10,
                                    device=gaussian_device, dtype=torch.float16)

        if not self.allow_tqdm and self.verbose:
            print(f'running prediction: {len(slicers)} steps')

        for p in self.list_of_parameters:
            # network weights have to be updated outside autocast!
            # we are loading parameters on demand instead of loading them upfront. This reduces memory footprint a lot.
            # each set of parameters is only used once on the test set (one image) so run time wise this is almost the
            # same
            self.network.load_state_dict(torch.load(p, map_location=compute_device, weights_only=False)['network_weights'])
            with torch.autocast(self.device.type, enabled=True):
                for sl in tqdm(slicers, disable=not self.allow_tqdm):
                    pred = self._internal_maybe_mirror_and_predict(data[sl][None].to(compute_device))[0].to(
                        gaussian_device)
                    pred /= (pred.max() / 100)
                    pred = torch.nn.functional.one_hot((pred * gaussian).argmax(0), num_classes=self.label_manager.num_segmentation_heads).permute(3, 0, 1, 2).to(
                        predicted_logits_device)
                    predicted_logits[sl][:46] += pred[:46]
                    predicted_logits[sl][46] += pred[46:].sum(0)

                del pred
        empty_cache(self.device)
        return predicted_logits

    def convert_predicted_logits_to_segmentation_with_correct_shape(self, predicted_logits, props):
        old = torch.get_num_threads()
        torch.set_num_threads(7)

        # # resample to original shape
        # spacing_transposed = [props['spacing'][i] for i in self.plans_manager.transpose_forward]
        # current_spacing = self.configuration_manager.spacing if \
        #     len(self.configuration_manager.spacing) == \
        #     len(props['shape_after_cropping_and_before_resampling']) else \
        #     [spacing_transposed[0], *self.configuration_manager.spacing]
        # predicted_logits = self.configuration_manager.resampling_fn_probabilities(predicted_logits,
        #                                                                           props[
        #                                                                               'shape_after_cropping_and_before_resampling'],
        #                                                                           current_spacing,
        #                                                                           [props['spacing'][i] for i in
        #                                                                            self.plans_manager.transpose_forward])

        segmentation = None
        pp = None
        try:
            with torch.no_grad():
                pp = predicted_logits.to('cuda:0')
                segmentation = pp.argmax(0).cpu()
                del pp
        except RuntimeError:
            del segmentation, pp
            torch.cuda.empty_cache()
            segmentation = predicted_logits.argmax(0)
        del predicted_logits

        # segmentation may be torch.Tensor but we continue with numpy
        if isinstance(segmentation, torch.Tensor):
            segmentation = segmentation.cpu().numpy()

        # put segmentation in bbox (revert cropping)
        segmentation_reverted_cropping = np.zeros(props['shape_before_cropping'],
                                                  dtype=np.uint8 if len(
                                                      self.label_manager.foreground_labels) < 255 else np.uint16)
        slicer = bounding_box_to_slice(props['bbox_used_for_cropping'])
        segmentation_reverted_cropping[slicer] = segmentation
        del segmentation

        # revert transpose
        segmentation_reverted_cropping = segmentation_reverted_cropping.transpose(self.plans_manager.transpose_backward)
        torch.set_num_threads(old)
        return segmentation_reverted_cropping


def predict_semseg(im, prop, semseg_trained_model, semseg_folds, tile_step_size):
    # initialize predictors
    pred_semseg = CustomPredictor(
        tile_step_size=tile_step_size,
        use_mirroring=True,
        use_gaussian=True,
        perform_everything_on_device=False,
        allow_tqdm=True
    )
    pred_semseg.initialize_from_trained_model_folder(
        semseg_trained_model,
        use_folds=semseg_folds,
        checkpoint_name='checkpoint_final.pth'
    )

    semseg_pred = pred_semseg.predict_single_npy_array(
        im, prop, None
    )
    torch.cuda.empty_cache()
    gc.collect()
    return semseg_pred


def map_labels_to_toothfairy(predicted_seg: np.ndarray) -> np.ndarray:
    # Create an array that maps the labels directly
    max_label = 77
    mapping = np.arange(max_label + 1)

    # Define the specific remapping
    remapping = {19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 26, 25: 27, 26: 28,
                 27: 31, 28: 32, 29: 33, 30: 34, 31: 35, 32: 36, 33: 37, 34: 38,
                 35: 41, 36: 42, 37: 43, 38: 44, 39: 45, 40: 46, 41: 47, 42: 48,
                 43: 51, 44: 52, 45: 53,
                 46: 50, 47: 50, 48: 50, 49: 50, 50: 50, 51: 50, 52: 50, 53: 50, 
                 54: 50, 55: 50, 56: 50, 57: 50, 58: 50, 59: 50, 60: 50, 61: 50,
                 62: 50, 63: 50, 64: 50, 65: 50, 66: 50, 67: 50, 68: 50, 69: 50,
                 70: 50, 71: 50, 72: 50, 73: 50, 74: 50, 75: 50, 76: 50, 77: 50}
                #  46: 111, 47: 112, 48: 113, 49: 114, 50: 115, 51: 116, 52: 117, 53: 118,
                #  54: 121, 55: 122, 56: 123, 57: 124, 58: 125, 59: 126, 60: 127, 61: 128,
                #  62: 131, 63: 132, 64: 133, 65: 134, 66: 135, 67: 136, 68: 137, 69: 138,
                #  70: 141, 71: 142, 72: 143, 73: 144, 74: 145, 75: 146, 76: 147, 77: 148,}
    

    # Apply the remapping
    for k, v in remapping.items():
        mapping[k] = v

    return mapping[predicted_seg]


def postprocess(prediction_npy, vol_per_voxel, verbose: bool = False):
    cutoffs = {1: 0.0,
            #    2: 78411.5,
               2: 100.0,
               3: 0.0,
               4: 0.0,
               5: 204.0,
               6: 331.0,
               7: 0.0,
               8: 3028.0,
               9: 9798.0,
               10: 946.0,
               11: 1143.0,
               12: 676.0,
               13: 1100,
               14: 320,
               15: 1267.0,
               16: 1751.0,
               17: 1819.0,
               18: 1452.5,
               19: 1420.0,
               20: 1000.0,
               21: 536.0,
               22: 712.0,
               23: 469.0,
               24: 756.0,
               25: 1690.0,
               26: 1584.0,
               27: 0.0,
               28: 0.0,
               29: 0.0,
               30: 0.0,
               31: 1935.5,
               32: 0.0,
               33: 0.0,
               34: 6140.0,
               35: 0.0,
               36: 0.0,
               37: 0.0,
               38: 2710.0,
               39: 0.0,
               40: 0.0,
               41: 0.0,
               42: 970.0}

    for c in cutoffs.keys():
        co = cutoffs[c]
        if co > 0:
            mask = prediction_npy == c
            pred_vol = np.sum(mask)
            if 0 < pred_vol < co :
                prediction_npy[mask] = 0
                if verbose:
                    print(
                        f'removed label {c} because predicted volume of {pred_vol} is less than the cutoff {co}')
    return prediction_npy

def clip_seg(predicted_seg: np.ndarray) -> np.ndarray:
    return np.clip(predicted_seg, 0, 53)

def warp_post(semseg_pred, labels):
    # tmp_seg = np.zeros_like(semseg_pred)
    # labels = [i for i in range(1, 46)]
    semseg_pred = remove_all_but_largest_component_from_segmentation(semseg_pred, labels)
    return semseg_pred

def remove_small(semseg_pred, labels, value=3000):
    for i in labels:
        if np.sum(semseg_pred == i) < value:
            semseg_pred[semseg_pred == i] = 0
    return semseg_pred

def opening(semseg_pred, structure=None,label=7):

    mask = semseg_pred==label
    if np.sum(mask)==0:
        return semseg_pred
    if structure is None:
        structure = np.ones((3, 3, 3), dtype=int)
        
    # Erosion followed by dilation
    eroded_mask = binary_erosion(mask, structure=structure).astype(int)
    opened_mask = binary_dilation(eroded_mask, structure=structure).astype(int)
    
    semseg_pred[semseg_pred==label]=0
    semseg_pred[opened_mask==1]=label

    return semseg_pred

def closing(semseg_pred, structure=None,label=45):

    mask = semseg_pred==label
    if np.sum(mask)==0:
        return semseg_pred
    if structure is None:
        structure = np.ones((3, 3, 3), dtype=int)
    
    # Closing operation: dilation followed by erosion
    dilated_mask = binary_dilation(mask, structure=structure).astype(int)
    closed_mask = binary_erosion(dilated_mask, structure=structure).astype(int)
    
    semseg_pred[semseg_pred==label]=0
    semseg_pred[closed_mask==1]=label

    return semseg_pred

def iac_pred(input_image_file_path,semseg_pred,debug=False):
    if input_image_file_path.endswith('.mha'):   
        input_image = sitk.ReadImage(input_image_file_path)
        sitk.WriteImage(input_image, 'image.nii.gz')
        input_image_file_path = 'image.nii.gz'
    from iac import InferClass
    infer = InferClass() if not debug else InferClass(config_file='./iac/configs/infer_iac_debug.yaml')
    output_array = infer.infer_everything(input_image_file_path, label_prompt=[133,134],save_mask=False)
    output_array = output_array[0].transpose(2, 0)
    semseg_pred[semseg_pred==3] = 0
    semseg_pred[semseg_pred==4] = 0
    semseg_pred[output_array==133] = 3
    semseg_pred[output_array==134] = 4
    return semseg_pred

def filter_pulp(semseg_pred):
    classed = np.unique(semseg_pred)
    
    for i in classed:
        if i < 78 and i>45 and (i-35) not in classed:
            semseg_pred[semseg_pred==i] = 0
            print('removed class',i)
    
    return semseg_pred
    

if __name__ == '__main__':
    os.environ['nnUNet_compile'] = 'f'

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', type=Path, default="/input/images/cbct/")
    parser.add_argument('-o', '--output_folder', type=Path, default="/output/images/oral-pharyngeal-segmentation/")
    parser.add_argument('-sem_mod', '--semseg_trained_model', type=str,
                        default="/opt/ml/model/nnUNetTrainer_onlyMirror01_1500ep__nnUNetResEncUNetLPlans_torchres__3d_fullres_torchres_ps128x224x224_bs2")


    parser.add_argument('--semseg_folds', type=str, nargs='+', default=['all'])
    args = parser.parse_args()

    args.output_folder.mkdir(exist_ok=True, parents=True)

    semseg_folds = [i if i == 'all' else int(i) for i in args.semseg_folds]
    semseg_trained_model = args.semseg_trained_model
    
    rw = SimpleITKIO()

    input_files = list(args.input_folder.glob('*.nii.gz')) + list(args.input_folder.glob('*.mha'))

    for input_fname in input_files:
        output_fname = args.output_folder / input_fname.name

        # we start with the instance seg because we can then start converting that while semseg is being predicted
        # load test image
        im, prop = rw.read_images([input_fname])
        tile_step_size = 0.5
        slices = compute_steps_for_sliding_window(im.shape[1:], [128, 224, 224], tile_step_size)
        size = len(slices[0])*len(slices[1])*len(slices[2])

        while size < 20:
            tile_step_size -= 0.05
            slices = compute_steps_for_sliding_window(im.shape[1:], [128, 224, 224], tile_step_size)
            size = len(slices[0])*len(slices[1])*len(slices[2])

        with torch.no_grad():
            semseg_pred = predict_semseg(im, prop, semseg_trained_model, semseg_folds, tile_step_size)
            torch.cuda.empty_cache()
            gc.collect()

        # now postprocess
        semseg_pred = postprocess(semseg_pred, np.prod(prop['spacing']), True)
        semseg_pred = iac_pred(str(input_fname),semseg_pred,not semseg_trained_model.startswith('/opt/ml'))

        print(np.unique(semseg_pred))
        labels = [1, 7] + \
                [j for j in range(11, 43)] 

        semseg_pred = opening(semseg_pred,label=2)
        semseg_pred = opening(semseg_pred,label=7)

        for i in labels:
            semseg_pred = warp_post(semseg_pred, [i]) 
            
               
        semseg_pred = map_labels_to_toothfairy(semseg_pred)

        # now save
        rw.write_seg(semseg_pred, output_fname, prop)
