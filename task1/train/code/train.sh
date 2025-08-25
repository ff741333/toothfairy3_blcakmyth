export nnUNet_raw='xxx'
export nnUNet_preprocessed="xxx"
export nnUNet_results="xxx"
export nnUNet_n_proc_DA=32


CUDA_VISIBLE_DEVICES=2,3 nnUNetv2_train 113 3d_fullres_torchres_ps128x224x224_bs2 all -p nnUNetResEncUNetLPlans_torchres -tr nnUNetTrainer_onlyMirror01_1500ep -num_gpus 2 --val
