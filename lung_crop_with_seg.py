import os
import time
import numpy as np
import SimpleITK as sitk
from skimage import measure
from scipy import ndimage


def get_ND_bounding_box(volume, margin=None):
    """
    Get the bounding box of nonzero region in an ND volume.
    :param volume: An ND numpy array.
    :param margin: (list)
        The margin of bounding box along each axis.
    :return bb_min: (list) A list for the minimal value of each axis
            of the bounding box.
    :return bb_max: (list) A list for the maximal value of each axis
            of the bounding box.
    """
    input_shape = volume.shape
    if (margin is None):
        margin = [0] * len(input_shape)
    assert (len(input_shape) == len(margin))
    indxes = np.nonzero(volume)
    bb_min = []
    bb_max = []
    for i in range(len(input_shape)):
        bb_min.append(int(indxes[i].min()))
        bb_max.append(int(indxes[i].max()) + 1)

    for i in range(len(input_shape)):
        bb_min[i] = max(bb_min[i] - margin[i], 0)
        bb_max[i] = min(bb_max[i] + margin[i], input_shape[i])
    return bb_min, bb_max


def crop_ND_volume_with_bounding_box(volume, bb_min, bb_max):
    """
    Extract a subregion form an ND image.
    :param volume: The input ND array.
    :param bb_min: (list) The lower bound of the bounding box for each axis.
    :param bb_max: (list) The upper bound of the bounding box for each axis.
    :return: A croped ND image.
    """
    dim = len(volume.shape)
    assert (dim >= 2 and dim <= 5)
    assert (bb_max[0] - bb_min[0] <= volume.shape[0])
    if (dim == 2):
        output = volume[bb_min[0]:bb_max[0], bb_min[1]:bb_max[1]]
    elif (dim == 3):
        output = volume[bb_min[0]:bb_max[0], bb_min[1]:bb_max[1], bb_min[2]:bb_max[2]]
    elif (dim == 4):
        output = volume[bb_min[0]:bb_max[0], bb_min[1]:bb_max[1], bb_min[2]:bb_max[2], bb_min[3]:bb_max[3]]
    elif (dim == 5):
        output = volume[bb_min[0]:bb_max[0], bb_min[1]:bb_max[1], bb_min[2]:bb_max[2], bb_min[3]:bb_max[3],
                 bb_min[4]:bb_max[4]]
    else:
        raise ValueError("the dimension number shoud be 2 to 5")
    return output


def get_largest_k_components(image, k=1):
    """
    Get the largest K components from 2D or 3D binary image.
    :param image: The input ND array for binary segmentation.
    :param k: (int) The value of k.
    :return: An output array (k == 1) or a list of ND array (k>1)
        with only the largest K components of the input.
    """
    dim = len(image.shape)
    if image.sum() == 0:
        print('the largest component is null')
        return image
    if dim < 2 or dim > 3:
        raise ValueError("the dimension number should be 2 or 3")
    s = ndimage.generate_binary_structure(dim, 1)
    labeled_array, numpatches = ndimage.label(image, s)
    sizes = ndimage.sum(image, labeled_array, range(1, numpatches + 1))
    sizes_sort = sorted(sizes, reverse=True)
    kmin = min(k, numpatches)
    output = []
    for i in range(kmin):
        labeli = min(np.where(sizes == sizes_sort[i])[0]) + 1
        output_i = np.asarray(labeled_array == labeli, np.uint8)
        output.append(output_i)
    return output[0] if k == 1 else output


def crop_ct_scan(input_img, input_seg):
    """
    Crop a CT scan based on the bounding box of the human region.
    """
    img = sitk.GetArrayFromImage(input_img)
    seg = sitk.GetArrayFromImage(input_seg)

    mask = np.asarray(img > -600)
    se = np.ones([3, 3, 3])
    mask = ndimage.binary_opening(mask, se, iterations=2)
    mask = get_largest_k_components(mask, 1)
    bbmin, bbmax = get_ND_bounding_box(mask, margin=[5, 10, 10])

    origin = input_img.GetOrigin()
    spacing = input_img.GetSpacing()
    new_origin = tuple([origin[i] + spacing[i] * bbmin[::-1][i] for i in range(len(bbmin))])

    img_sub = crop_ND_volume_with_bounding_box(img, bbmin, bbmax)
    img_sub_obj = sitk.GetImageFromArray(img_sub)
    img_sub_obj.SetOrigin(new_origin)
    img_sub_obj.SetSpacing(spacing)
    img_sub_obj.SetDirection(input_img.GetDirection())

    seg_sub = crop_ND_volume_with_bounding_box(seg, bbmin, bbmax)
    seg_sub_obj = sitk.GetImageFromArray(seg_sub)
    seg_sub_obj.SetOrigin(new_origin)
    seg_sub_obj.SetSpacing(spacing)
    seg_sub_obj.SetDirection(input_img.GetDirection())

    return img_sub_obj, seg_sub_obj


def get_human_region_mask(img):
    """
    Get the mask of human region in CT volumes
    """
    dim = len(img.shape)
    if dim == 4:
        img = img[0]
    mask = np.asarray(img > -600)
    se = np.ones([3, 3, 3])
    mask = ndimage.binary_opening(mask, se, iterations=2)
    mask = get_largest_k_components(mask, 1)
    mask_close = ndimage.binary_closing(mask, se, iterations=2)

    D, H, W = mask.shape
    for d in [1, 2, D - 3, D - 2]:
        mask_close[d] = mask[d]
    for d in range(0, D, 2):
        mask_close[d, 2:-2, 2:-2] = np.ones((H - 4, W - 4))

    # get background component
    bg = np.zeros_like(mask)
    bgs = get_largest_k_components(1 - mask_close, 10)
    for bgi in bgs:
        indices = np.where(bgi)
        if bgi.sum() < 1000:
            break
        if indices[0].min() == 0 or indices[1].min() == 0 or indices[2].min() == 0 or indices[0].max() == D - 1 or \
                indices[1].max() == H - 1 or indices[2].max() == W - 1:
            bg = bg + bgi
    fg = 1 - bg

    fg = ndimage.binary_opening(fg, se, iterations=1)
    fg = get_largest_k_components(fg, 1)
    if dim == 4:
        fg = np.expand_dims(fg, 0)
    fg = np.asarray(fg, np.uint8)
    return fg


def lungmask(vol):
    # 获取体数据的尺寸
    size = sitk.Image(vol).GetSize()
    # 获取体数据的空间尺寸
    spacing = sitk.Image(vol).GetSpacing()
    # 将体数据转为numpy数组
    volarray = sitk.GetArrayFromImage(vol)

    # 根据CT值，将数据二值化（一般来说-300以下是空气的CT值）
    volarray[volarray >= -300] = 1
    volarray[volarray <= -300] = 0
    # 生成阈值图像
    # threshold = ndimage.binary_opening(volarray, iterations=1).astype("uint8")
    threshold = sitk.GetImageFromArray(volarray)
    threshold.SetSpacing(spacing)

    bodymask = sitk.GetImageFromArray(get_human_region_mask(sitk.GetArrayFromImage(vol)))
    bodymask.SetSpacing(spacing)

    # 用bodymask减去threshold，得到初步的lung的mask
    temp = sitk.GetArrayFromImage(bodymask) - sitk.GetArrayFromImage(threshold)
    temp[temp != 1] = 0
    temp = ndimage.binary_opening(temp, iterations=1).astype("uint8")
    temp = sitk.GetImageFromArray(temp)
    temp.SetSpacing(spacing)

    # 利用形态学来去掉一定的肺部的小区域
    bm = sitk.BinaryMorphologicalClosingImageFilter()
    bm.SetKernelType(sitk.sitkBall)
    bm.SetKernelRadius(2)
    bm.SetForegroundValue(1)
    lungmask = bm.Execute(temp)

    # 利用measure来计算连通域
    lungmaskarray = sitk.GetArrayFromImage(lungmask)
    core_dilation = np.ones([7, 7, 7])
    lungmaskarray = ndimage.binary_dilation(lungmaskarray, core_dilation, iterations=1).astype("uint8")
    core_opening = np.ones([3, 3, 3])
    lungmaskarray = ndimage.binary_opening(lungmaskarray, core_opening, iterations=1).astype("uint8")
    label = measure.label(lungmaskarray, connectivity=2)
    props = measure.regionprops(label)

    # 计算每个连通域的体素的个数
    numPix = []
    for ia in range(len(props)):
        numPix += [props[ia].area]

    # 最大连通域的体素个数，也就是肺部
    maxnum = max(numPix)
    # 遍历每个连通区域
    for i in range(len(numPix)):
        # 如果当前连通区域不是最大值所在的区域，则当前区域的值全部置为0，否则为1
        if numPix[i] != maxnum:
            label[label == i + 1] = 0
        else:
            label[label == i + 1] = 1

    return label.astype("uint8")


def crop_to_lung_area(file_path, img_save_path, seg_path, seg_save_path):
    vol = sitk.ReadImage(file_path)
    seg = sitk.ReadImage(seg_path)
    crop_to_body, seg_crop_to_body = crop_ct_scan(vol, seg)
    # sitk.WriteImage(crop_to_body, "/home/wlty/disk2/nnUNet_data/nnUNet_raw/Dataset110_CTLymphNodes/body.nii.gz")
    seg_array = sitk.GetArrayFromImage(seg_crop_to_body)

    mask = lungmask(crop_to_body)
    # mask_obj = sitk.GetImageFromArray(mask)
    # mask_obj.CopyInformation(crop_to_body)
    # sitk.WriteImage(mask_obj, "/home/wlty/disk2/nnUNet_data/nnUNet_raw/Dataset110_CTLymphNodes/mask.nii.gz")

    bbmin = [0, 0, 0]
    bbmax = [0, 0, 0]
    bbmin_img, bbmax_img = get_ND_bounding_box(mask)
    # print(bbmin_img, bbmax_img)
    bbmin_seg, bbmax_seg = get_ND_bounding_box(seg_array, margin=(5, 10, 10))
    # print(bbmin_seg, bbmax_seg)
    for i in range(len(bbmin_img)):
        bbmin[i] = min(bbmin_img[i], bbmin_seg[i])
        bbmax[i] = max(bbmax_img[i], bbmax_seg[i])
        
    crop_shape = sitk.GetArrayFromImage(crop_to_body).shape
    center = np.array(crop_shape) // 2
    # print(center)
    
    for i in range(1, 3):
        if bbmin[i] < center[i] < bbmax[i]:
            if (center[i] - bbmin[i])/(bbmax[i] - center[i]) >= 2:
                bbmax[i] = crop_shape[i] - bbmin[i]
            elif (center[i] - bbmin[i])/(bbmax[i] - center[i]) <= 0.5:
                bbmin[i] = crop_shape[i] - bbmax[i]
        elif bbmin[i] >= center[i]:
            bbmin[i] = crop_shape[i] - bbmax[i]
        elif bbmax[i] <= center[i]:
            bbmax[i] = crop_shape[i] - bbmin[i]

    origin = vol.GetOrigin()
    spacing = vol.GetSpacing()
    origin_output = tuple([origin[i] + spacing[i] * bbmin[::-1][i] for i in range(len(bbmin))])
    img_output = crop_ND_volume_with_bounding_box(sitk.GetArrayFromImage(crop_to_body), bbmin, bbmax)
    seg_output = crop_ND_volume_with_bounding_box(seg_array, bbmin, bbmax)

    img_output = sitk.GetImageFromArray(img_output)
    img_output.SetOrigin(origin_output)
    img_output.SetSpacing(spacing)
    img_output.SetDirection(vol.GetDirection())
    sitk.WriteImage(img_output, img_save_path)

    seg_output = sitk.GetImageFromArray(seg_output)
    seg_output.SetOrigin(origin_output)
    seg_output.SetSpacing(spacing)
    seg_output.SetDirection(vol.GetDirection())
    sitk.WriteImage(seg_output, seg_save_path)


if __name__ == "__main__":
    img_folder_path = "./imagesTr1"
    seg_folder_path = "./labelsTr1"
    img_save_folder = "./imagesTr"
    seg_save_folder = "./labelsTr"
    if not os.path.isdir(img_save_folder):
        os.mkdir(img_save_folder)
    if not os.path.isdir(seg_save_folder):
        os.mkdir(seg_save_folder)

    file_lst = os.listdir(img_folder_path)
    length = len(file_lst)
    i = 1
    start = time.time()
    for file in file_lst:
        print("{}/{} Processing: {}".format(i, length, file))
        i += 1
        idx = file.split("_")[1]
        file_path = os.path.join(img_folder_path, file)
        img_save_path = os.path.join(img_save_folder, file)
        seg_filename = "LNQ2023_{}.nrrd".format(idx)
        seg_path = os.path.join(seg_folder_path, seg_filename)
        seg_save_path = os.path.join(seg_save_folder, seg_filename)

        start_case = time.time()
        crop_to_lung_area(file_path, img_save_path, seg_path, seg_save_path)
        print("Cost: {:.2f}s".format(time.time() - start_case))

    print("AVG Processing Cost: {:.2f}s".format((time.time() - start) / len(file_lst)))
