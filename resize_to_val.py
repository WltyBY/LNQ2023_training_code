import os
import numpy as np
import SimpleITK as sitk


def trans_to_ori(img_path, seg_path, output_path):
    # there are some samples whose img and seg have different size, use this to transform
    img_obj = sitk.ReadImage(img_path)
    seg_obj = sitk.ReadImage(seg_path)
    seg_array = sitk.GetArrayFromImage(seg_obj)

    direction = img_obj.GetDirection()

    spacing = img_obj.GetSpacing()
    origin_ori = np.array(img_obj.GetOrigin())
    origin_now = np.array(seg_obj.GetOrigin())
    # size_ori = np.array(img_obj.GetSize())
    # size_now = np.array(seg_obj.GetSize())
    seg_output = np.zeros_like(sitk.GetArrayFromImage(img_obj))
    # index_min = origin_now.astype(np.int16)[::-1]
    # index_max = (index_min + seg_array.shape).astype(np.int16)
    index_min = np.round((origin_now - origin_ori) / spacing).astype("int32")
    index_max = (index_min + seg_array.shape).astype("int32")
    seg_output[index_min[0]:index_max[0], index_min[1]:index_max[1], index_min[2]:index_max[2]] = seg_array

    seg_output = sitk.GetImageFromArray(seg_output.astype("uint8"))
    seg_output.SetOrigin(origin_ori)
    seg_output.SetSpacing(spacing)
    seg_output.SetDirection(direction)
    sitk.WriteImage(seg_output, output_path)


if __name__ == "__main__":
    img_folder = "./val"
    seg_folder = "./postprocessed_prevnetv2_200"
    output_folder = "./output_prevnetv2_200"
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    img_file_lst = os.listdir(img_folder)
    for img_file in img_file_lst:
        img_file_path = os.path.join(img_folder, img_file)
        seg_file_path = os.path.join(seg_folder, "lnq2023-val-{}-seg.nrrd".format(img_file[12:16]))
        output_path = os.path.join(output_folder, "lnq2023-val-{}-seg.nrrd".format(img_file[12:16]))
        trans_to_ori(img_file_path, seg_file_path, output_path)
