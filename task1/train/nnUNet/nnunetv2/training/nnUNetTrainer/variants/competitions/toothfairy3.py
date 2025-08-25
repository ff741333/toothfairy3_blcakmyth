import torch
from nnunetv2.training.nnUNetTrainer.variants.wnet.nnUNetTrainer_WNet3D import nnUNetTrainer_WNet3D
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerNoMirroring import nnUNetTrainer_onlyMirror01

class nnUNetTrainer_onlyMirror01_WNet3D_1500ep(nnUNetTrainer_onlyMirror01,nnUNetTrainer_WNet3D):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1500
