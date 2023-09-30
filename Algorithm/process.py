import torch
import os
import SimpleITK
import numpy as np

from pathlib import Path
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from lung_crop_without_seg import crop2lung
from process_utils import label_refine
from resize_to_val import trans2ori

from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)


class Lnq2023(SegmentationAlgorithm):
    def __init__(self):
        output_path = Path('/output/images/mediastinal-lymph-node-segmentation/')
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
        super().__init__(
            input_path=Path('/input/images/mediastinal-ct/'),
            output_path=output_path,
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )
        self.predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_gpu=True,
            device=torch.device('cuda', 0),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True
        )
        self.predictor.initialize_from_trained_model_folder(
            os.path.join("/opt/app/nnUNet_results", 'Dataset081_LNQ2023/nnUNetPreTrainerVNetv2__nnUNetPlans__3d_fullres'),
            use_folds=('all', ),
            checkpoint_name='checkpoint_final.pth',
        )

    def predict(self, *, input_image: SimpleITK.Image) -> SimpleITK.Image:
        # TODO: add your algorithm here
        # Crop to ROI
        img_after_crop = crop2lung(input_image)

        props = {
            'sitk_stuff': {
                # this saves the sitk geometry information. This part is NOT used by nnU-Net!
                'spacing': img_after_crop.GetSpacing(),
                'origin': img_after_crop.GetOrigin(),
                'direction': img_after_crop.GetDirection()
            },
            # the spacing is inverted with [::-1] because sitk returns the spacing in the wrong order lol. Image arrays
            # are returned x,y,z but spacing is returned z,y,x. Duh.
            'spacing': list(img_after_crop.GetSpacing())[::-1]
        }
        npy_image = SimpleITK.GetArrayFromImage(img_after_crop).astype(np.float32)
        ret = self.predictor.predict_single_npy_array(npy_image[None], props, None, None, False)
        
        # iterator = self.predictor.get_data_iterator_from_raw_npy_data([npy_image], None, [props], None, 1)
        # ret = self.predictor.predict_from_data_iterator(iterator, False, 1)

        # label refine
        vol_per_pixel = np.prod(np.array(props['spacing']))
        seg_array = label_refine(npy_image, ret, vol_per_pixel)

        #resize to original size
        seg_output = trans2ori(input_image, seg_array, props)

        return seg_output


if __name__ == "__main__":
    Lnq2023().process()
