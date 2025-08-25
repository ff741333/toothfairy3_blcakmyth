export nnUNet_raw='xxx'
export nnUNet_preprocessed="xxx"
export nnUNet_results="xxx"
export nnUNet_n_proc_DA=32

nnUNetv2_plan_and_preprocess -d 113 -pl nnUNetPlannerResEncL_torchres
nnUNetv2_preprocess -d 113 -c 3d_fullres_torchres_ps128x224x224_bs2 -plans_name nnUNetResEncUNetLPlans_torchres -np 48