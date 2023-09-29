import os
import shutil
import SimpleITK as sitk
from nnunetv2.paths import nnUNet_raw
from nnunetv2.utilities.dataset_name_id_conversion import find_candidate_datasets
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, subfiles


def trans_seg_size_as_img(img_path, seg_path, output_path):
    # there are some samples whose img and seg have different size, use this to transform
    img_obj = sitk.ReadImage(img_path)
    seg_obj = sitk.ReadImage(seg_path)

    spacing = img_obj.GetSpacing()
    origin = img_obj.GetOrigin()
    direction = img_obj.GetDirection()
    input_size = img_obj.GetSize()

    output_spacing = spacing[:]
    output_size = input_size[:]
    transform = sitk.Transform(3, sitk.sitkIdentity)
    resampled_img = sitk.Resample(seg_obj, output_size, transform, sitk.sitkLinear, origin, output_spacing, direction)
    sitk.WriteImage(resampled_img, output_path)


def convert_LNQ_dataset(source_folder, overwrite_target_id=None):
    if source_folder.endswith('/') or source_folder.endswith('\\'):
        source_folder = source_folder[:-1]

    train = os.path.join(source_folder, 'train')
    assert os.path.isdir(train), f"train subfolder missing in source folder"
    if len(os.listdir(source_folder)) == 2:
        # only have two folders: train set and train labels in the folder named train, val set in the folder named val
        val = os.path.join(source_folder, 'val')
        assert os.path.isdir(val), f"val subfolder missing in source folder"
    elif len(os.listdir(source_folder)) == 3:
        # have folders: train, val and test
        test = os.path.join(source_folder, 'test')
        assert os.path.isdir(test), f"test subfolder missing in source folder"

    # infer source dataset id and name
    # "Task080_LNQ2023" -> "Task080", "LNQ2023"
    task, dataset_name = os.path.basename(source_folder).split('_')
    task_id = int(task[4:])    # 80(int)

    # check if target dataset id is taken
    target_id = task_id if overwrite_target_id is None else overwrite_target_id
    existing_datasets = find_candidate_datasets(target_id)
    assert len(existing_datasets) == 0, f"Target dataset id {target_id} is already taken, please consider changing " \
                                        f"it using overwrite_target_id. Conflicting dataset: {existing_datasets} (check nnUNet_results, nnUNet_preprocessed and nnUNet_raw!)"

    target_dataset_name = f"Dataset{target_id:03d}_{dataset_name}"
    target_folder = os.path.join(nnUNet_raw, target_dataset_name)
    target_imagesTr = os.path.join(target_folder, 'imagesTr1')
    target_labelsTr = os.path.join(target_folder, 'labelsTr1')
    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_labelsTr)
    if len(os.listdir(source_folder)) in [2, 3]:
        target_imagesTs = os.path.join(target_folder, 'imagesTs')
        maybe_mkdir_p(target_imagesTs)

    # copy files to target folder
    len_val_set = 0
    len_test_set = 0

    source_images = [i for i in subfiles(train, suffix='-ct.nrrd', join=False) if
                     not i.startswith('.') and not i.startswith('_')]
    for s in source_images:
        img_path = os.path.join(train, s)
        seg_path = os.path.join(train, "lnq2023-train-{}-seg.nrrd".format(s[14:18]))
        seg_output_path = os.path.join(target_labelsTr, "LNQ2023_{}.nrrd".format(s[14:18]))

        # copy train images
        shutil.copy(img_path, os.path.join(target_imagesTr, "LNQ2023_{}_0000.nrrd".format(s[14:18])))

        # copy train segmentations or transform seg if needed
        img_size = sitk.ReadImage(img_path).GetSize()
        seg_size = sitk.ReadImage(seg_path).GetSize()
        print("Sample:", s[14:18])
        print("Img_size:", img_size)
        print("Seg_size", seg_size)
        if img_size == seg_size:
            shutil.copy(seg_path, seg_output_path)
        else:
            print("Trans!")
            trans_seg_size_as_img(img_path, seg_path, seg_output_path)
    len_train_set = len(source_images)


    source_images = [i for i in subfiles(train, suffix='-seg.nrrd', join=False) if
                     not i.startswith('.') and not i.startswith('_')]
    len_train_labels = len(source_images)

    # whether we have val or test images(the images we use our model to infer)
    if len(os.listdir(source_folder)) == 2:
        source_images = [i for i in subfiles(val, suffix='-ct.nrrd', join=False) if
                         not i.startswith('.') and not i.startswith('_')]
        for s in source_images:
            shutil.copy(os.path.join(val, s), os.path.join(target_imagesTs, "LNQ2023_{}_0000.nrrd".format(s[12:16])))
        len_val_set = len(source_images)

    elif len(os.listdir(source_folder)) == 3:
        source_images = [i for i in subfiles(val, suffix='-ct.nrrd', join=False) if
                         not i.startswith('.') and not i.startswith('_')]
        for s in source_images:
            shutil.copy(os.path.join(val, s), os.path.join(target_imagesTs, "LNQ2023_{}_0000.nrrd".format(s[12:16])))
        len_val_set = len(source_images)

        source_images = [i for i in subfiles(test, suffix='-ct.nrrd', join=False) if
                             not i.startswith('.') and not i.startswith('_')]
        for s in source_images:
            shutil.copy(os.path.join(test, s), os.path.join(target_imagesTs, "LNQ2023_{}_0000.nrrd".format(s[14:18])))
        len_test_set = len(source_images)

    len_imagesTr = len(os.listdir(target_imagesTr))
    len_labelsTr = len(os.listdir(target_labelsTr))
    assert len_imagesTr == len_train_set, "Number of train set error!"
    assert len_labelsTr == len_train_labels, "Number of train labels error!"
    assert len_labelsTr == len_imagesTr, "The number of train images isn't equal to the number of train labels!"
    if len(os.listdir(source_folder)) in [2, 3]:
        len_imagesTs = len(os.listdir(target_imagesTs))
        assert len_imagesTs == (len_val_set + len_test_set), "The number of val and test Error!"


if __name__ == "__main__":
    convert_LNQ_dataset("/hy-tmp/nnUNet_raw/Task080_LNQ2023")
