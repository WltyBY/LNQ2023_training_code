import os
import numpy as np
import SimpleITK as sitk
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from resize_to_val import trans_to_ori


# label_refine
def label_refine(img_array, seg_array, vol_per_pixel):
    MAX = 157 * 1.5
    MIN = -36 * 1.5
    MIN_VOL = 175.03343822511005
    
    seg_array = clear_border(seg_array, buffer_size=3)

    intensity_mask = (img_array <= MAX) & (img_array >= MIN)
    seg_array = intensity_mask * seg_array

    label_img = label(seg_array)
    regions = regionprops(label_img)

    for props in regions:
        num = props.num_pixels * vol_per_pixel
        label_id = props.label
        if num <= MIN_VOL:
            label_img[label_img == label_id] = 0
    label_img[label_img != 0] = 1

    return label_img.astype(np.uint8)


# resize_to_origin
def resize2origin_size(inputs_img_folder, inputs_seg_folder, outputs_folder):
    img_lst = os.listdir(inputs_img_folder)
    for img_file in img_lst:
        img_id = img_file.split("-")[2]
        img_file_path = os.path.join(inputs_img_folder, img_file)
        seg_file_path = os.path.join(inputs_seg_folder, "LNQ2023_{}.nrrd".format(img_id))
        output_path = os.path.join(outputs_folder, "lnq2023-val-{}-seg.nrrd".format(img_id))
        trans_to_ori(img_file_path, seg_file_path, output_path)
