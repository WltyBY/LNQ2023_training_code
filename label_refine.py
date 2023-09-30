import os
import numpy as np
import SimpleITK as sitk
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border


img_folder = "./val_crop"
seg_folder = "./val_prevnetv2_rebirth"
save_folder = "./postprocessed_prevnetv2_rebirth"
if not os.path.isdir(save_folder):
    os.mkdir(save_folder)

MAX = 157 * 1.5
MIN = -36 * 1.5
MIN_VOL = 175.03343822511005
img_files = os.listdir(img_folder)
i = 0
length = len(img_files)
for img_file in img_files:
    seg_file = "lnq2023-val-{}-seg.nrrd".format(img_file[12:16])
    i += 1
    print("{}/{} Postprocessing: {}".format(i, length, seg_file))
    seg_obj = sitk.ReadImage(os.path.join(seg_folder, seg_file))
    spacing = seg_obj.GetSpacing()
    vol_per_pixel = np.prod(spacing)
    origin = seg_obj.GetOrigin()
    direction = seg_obj.GetDirection()
    seg_array = sitk.GetArrayFromImage(seg_obj)
    seg_array = clear_border(seg_array, buffer_size=3)
    img_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(img_folder, img_file)))

    intensity_mask = (img_array <= MAX) & (img_array >= MIN)
    seg_array = intensity_mask * seg_array

    # seg_array = scipy.ndimage.binary_closing(seg_array, iterations=1)

    label_img = label(seg_array)
    regions = regionprops(label_img)

    for props in regions:
        num = props.num_pixels * vol_per_pixel
        label_id = props.label
        if num <= MIN_VOL:
            label_img[label_img == label_id] = 0

    label_img[label_img != 0] = 1
    seg_output = sitk.GetImageFromArray(label_img.astype(np.uint8))
    seg_output.SetOrigin(origin)
    seg_output.SetSpacing(spacing)
    seg_output.SetDirection(direction)
    sitk.WriteImage(seg_output, os.path.join(save_folder, seg_file))
