# Main Files

## Training file

Pretraining Trainer: [LNQ2023_training_code/nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetPreTrainer_PED.py at main · WltyBY/LNQ2023_training_code (github.com)](https://github.com/WltyBY/LNQ2023_training_code/blob/main/nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetPreTrainer_PED.py)

Weakly-supervised Trainer: [LNQ2023_training_code/nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetPreTrainerVNetv2.py at main · WltyBY/LNQ2023_training_code (github.com)](https://github.com/WltyBY/LNQ2023_training_code/blob/main/nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetPreTrainerVNetv2.py)

## Model setting

Model for training: [LNQ2023_training_code/nnUNet/nnunetv2/training/VNetv2.py at main · WltyBY/LNQ2023_training_code (github.com)](https://github.com/WltyBY/LNQ2023_training_code/blob/main/nnUNet/nnunetv2/training/VNetv2.py)

# Model's Weight And Docker

## Weight

Google Drive: https://drive.google.com/file/d/1L_ojs101tWh9XtGxAgE0pqceDlZhhaS8/view?usp=sharing

Baidu Netdisk: https://pan.baidu.com/s/1BrOhIUWXxEbyeMOpc5_O8w?pwd=0319

To load the weight, it is better to use function in [LNQ2023_training_code/nnUNet/nnunetv2/run/load_pretrained_weights.py at main · WltyBY/LNQ2023_training_code (github.com)](https://github.com/WltyBY/LNQ2023_training_code/blob/main/nnUNet/nnunetv2/run/load_pretrained_weights.py).

After postprocess, this weight got the results on validation set:

|        |   DSC(%)    | ASSD(mm)  |
| :----- | :---------: | :-------: |
| Weight | 54.50±19.84 | 7.59±5.83 |

## Docker

Grand Challenge: [lnq2023v1 - Grand Challenge (grand-challenge.org)](https://grand-challenge.org/algorithms/lnq2023v1/)

# Install nnUNet

Install the nnunetv2 by using

```bash
pip install -e .
```

**Note** that the nnunetv2 you should install is not that can be downloaded through the Internet. You should go into the nnUNet folder in this project and run the instruction above.

Assuming we set the environment variables as follows:

```bash
nnUNet_raw="/hy-tmp/nnUNet_raw"
nnUNet_preprocessed="/hy-tmp/nnUNet_preprocessed"
nnUNet_results="/hy-tmp/nnUNet_results"
```

You should follow you own settings!!

# Preprocess

Now, create a folder named **Task080_LNQ2023** in **nnUNet_raw** folder, then throw the two dataset folders(**train** and **val**) in it. Of course, you can set the name according to your preferences. Anyway, follow your own settings.

Then, run **convert_LNQ2023.py** in **nnUNet_raw** folder. This will trans the dataset into nnunet format. What's more, in this script, we will align samples with different sizes of images and labels based on the origin(This problem has been solved later by the organizers using **corrected_train_labels.zip** provided). Throw the **dataset.json** and **lung_crop_with_seg.py** into **Dataset080_LNQ2023**.

SO, your **Dataset080_LNQ2023** folder is as follows:

```
Dataset080_LNQ2023
 ├── imagesTr1
 ├── imagesTs
 ├── labelsTr1
 ├── lung_crop_with_seg.py
 └── dataset.json
```

run **crop_human_body_bbox.py**

```
Dataset080_LNQ2023
 ├── imagesTr
 ├── imagesTr1
 ├── imagesTs
 ├── labelsTr
 ├── labelsTr1
 ├── crop_human_body_bbox.py
 └── dataset.json
```

Run the following command

```bash
nnUNetv2_plan_and_preprocess -d 80 -c 3d_fullres --verify_dataset_integrity (-np 4)
```

Now, in your **nnUNet_preprocessed** folder, there will be a folder named **Dataset080_LNQ2023**. We can begin training.

# Self-Training

Run

```bash
nnUNetv2_train 80 3d_fullres all -tr=nnUNetPreTrainer_PED
```

where "all" means we will use all of our samples in training set to train the model and don't do K-fold cross-validation(K=5 in nnunet). We run this to use Model Genesis to train a pretrained weight.

During and after training, the model and log will be saved in **/hy-tmp/nnUNet_results/Dataset080_LNQ2023/nnUNetPreTrainer_PED_nnUNetPlans_3d_fullres/fold_all**.

# Weakly-Supervised Training

Before Training, we should load the pretrained weight.

So, run

```bash
nnUNetv2_train 80 3d_fullres all -tr=nnUNetPreTrainerVNetv2 -pretrained_weights=/hy-tmp/nnUNet_results/Dataset080_LNQ2023/nnUNetPreTrainer_PED_nnUNetPlans_3d_fullres/fold_all/checkpoint_best.pth
```

In /hy-tmp/nnUNet_results/Dataset080_LNQ2023/nnUNetPreTrainerVNetv2_nnUNetPlans_3d_fullres/fold_all folder, use checkpoint_final.pth as the result, because when doing training, the ground_truth is not accessible.

# Infer

**Note**: In order to save time, it would be better to preprocess the images inferred later. To do this, run **lung_crop_without_seg.py**.

```bash
nnUNetv2_predict -i data_path_after_preprocess_above -o output_save_path -d 80 -p nnUNetPlans -c 3d_fullres -f all -tr=nnUNetPreTrainerVNetv2
```

Then, run **rename_val_seg.py**, **label_refine.py**, **resize_to_val.py** in turn.

However, change the datas' paths in these scripts is needed.

# How to make a docker

In the **Algorithm** folder, the codes are written to process per case at one time, not like the infer way above which does per step all images together.

Before you use codes in this folder to create a docker, copy **nnUNet** folder I provide into **Algorithm** folder.

To run test.sh, your data should be in **/Algorithm/test/images/mediastinal-ct**, or you should change the path in test.sh.
