from monai.transforms.transform import MapTransform
from monai.transforms.croppad.dictionary import RandWeightedCropd, CropForegroundd, RandSpatialCropd, SpatialPadd, CenterSpatialCropd
from monai.transforms.intensity.dictionary import ScaleIntensityRangePercentilesd, NormalizeIntensityd
from monai.transforms.compose import Compose
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.utility.dictionary import EnsureChannelFirstd, DeleteItemsd, ToTensord, Lambdad
from monai.transforms.spatial.dictionary import Orientationd, Spacingd
from monai.transforms.intensity.dictionary import RandShiftIntensityd, RandScaleIntensityd
import numpy as np 
import torch

class SmartWeightedCrop(MapTransform):
    """
    Weighted crop if seg exists, otherwise random crop.
    Uses dictionary transforms to ensure same crop applied to all keys.
    """
    def __init__(self, keys, spatial_size, seg_key='seg'):
        super().__init__(keys)
        self.spatial_size = spatial_size
        self.seg_key = seg_key
        
        self.weighted_crop = RandWeightedCropd(
            keys=keys,
            w_key=seg_key,
            spatial_size=spatial_size,
            num_samples=1,
        )
        self.random_crop = RandSpatialCropd(
            keys=keys, 
            roi_size=spatial_size,
            random_size=False,
            allow_missing_keys=True
        )
    
    def __call__(self, data):
        d = dict(data)
        
        has_seg = self.seg_key in d and d[self.seg_key] is not None
        
        if has_seg:
            result = self.weighted_crop(d)
            print(len(result))
            d = result[0] if isinstance(result, list) else result # making the dict into a list for some reason
        else:
            d = self.random_crop(d)
        
        return d

class TumorCenterCrop(MapTransform):
    """
    Deterministic crop centered on tumor center of mass.
    Falls back to img center if no seg available.
    """
    def __init__(self, keys, spatial_size, seg_key='seg', image_key='image'):
        super().__init__(keys)
        self.spatial_size = np.array(spatial_size)
        self.seg_key = seg_key
        self.image_key = image_key
    
    def __call__(self, data):
        d = dict(data)
        
        has_seg = self.seg_key in d and d[self.seg_key] is not None
        
        if has_seg:
            # Get tumor center of mass
            seg = d[self.seg_key]

            tumor_mask = seg > 0 # assuming 0 is background, can change if we want to center on a specific tumor region
            
            if tumor_mask.sum() > 0:
                if tumor_mask.ndim == 4:
                    tumor_mask = tumor_mask[0]
                
                indices = np.argwhere(tumor_mask)
                center = indices.mean(axis=0).astype(int)
            else:
                center = np.array(d[self.image_key].shape[1:]) // 2
        else:
            # if no seg available, use image center
            center = np.array(d[self.image_key].shape[1:]) // 2
        
        # Apply crop centered on this point
        for key in self.keys:
            if key in d:
                d[key] = self._apply_center_crop(d[key], center)
        
        return d
    
    def _apply_center_crop(self, img, center):
        """
        Apply crop centered on given point
        """
        img_shape = np.array(img.shape[1:])  # Spatial dimensions
        
        # Calculate start and end
        half_size = self.spatial_size // 2
        start = center - half_size
        end = start + self.spatial_size
        
        # Ensure within bounds
        start = np.maximum(0, start)
        end = np.minimum(img_shape, end)
        
        for i in range(len(start)):
            if end[i] - start[i] < self.spatial_size[i]:
                if start[i] == 0:
                    end[i] = min(self.spatial_size[i], img_shape[i])
                elif end[i] == img_shape[i]:
                    start[i] = max(0, img_shape[i] - self.spatial_size[i])
        
        # Create slices
        slices = [slice(None)] 
        slices.extend([slice(int(s), int(e)) for s, e in zip(start, end)])
        
        return img[tuple(slices)]
    
def get_normalization_transform(config):
    """
    choice of normalization methods
    BrainSegFounder used NormalizeIntensity (Z-score)
    BrainMVP used ScaleIntensityRangePercentiles (Percentile)
    """
    norm_method = config["data"]["normalization_method"]

    if norm_method == "percentile":
        return ScaleIntensityRangePercentilesd(
            keys=['image'],
              lower=5, upper=95, b_min=0.0,
                b_max=1.0, 
                channel_wise=True)
    
    elif norm_method == "z_score":
        return NormalizeIntensityd(
            keys=["image"],
            nonzero=True, 
            channel_wise=True
            )
    else:
        raise ValueError(f"Unsupported normalization method {norm_method}")

def validate_transforms_config(config):

    # check the orientation axcodes
    valid_orientations = ["RAS", "LPS", "LAS", "RPS", "RAI", "LPI", "LAI", "RPI"]
    orientation = config["data"].get("orientation", "RAS")
    if orientation not in valid_orientations:
        raise ValueError(
            f"Invalid orientation: {orientation}"
            f"Must be one of {valid_orientations}"
        )
    config["data"]["orientation"] = orientation

    # check the roi types
    train_roi_type = config["data"].get("train_roi_type", None)
    if train_roi_type is None:
        print("No training ROI type provided in the config, falling back to random ROI")
        config["data"]["train_roi_type"] = "random"
    
    val_roi_type = config["data"].get("val_roi_type", None)
    if val_roi_type is None:
        print("No validation ROI type provided in the config, falling back to random ROI")
        config["data"]["val_roi_type"] = "random"

    # check roi sizes
    train_patch_shape = config["data"].get("train_patch_shape", None)
    if train_patch_shape is None:
        print("Train patch shape not provided, defaulting to 96^3")
        config["data"]["train_patch_shape"] = [96, 96, 96]

    val_patch_shape = config["data"].get("val_patch_shape", None)
    if val_patch_shape is None:
        print("No val patch shape provided, falling back to 160^3")
        config["data"]["val_patch_shape"] = [160, 160, 160]

    # norm method checked in function

def custom_transform(config):
    """
    Create custom transforms for training and validation
    """
    validate_transforms_config(config)

    train_roi_type = config["data"]["train_roi_type"]

    if train_roi_type == "random":
        # standard transforms if you dont have any segs to use
        print("Using random cropping for training")
        train_transform = Compose([
            LoadImaged(keys=['image']),
            EnsureChannelFirstd(keys=["image"]),
            get_normalization_transform(config),               
            Orientationd(keys=['image'], axcodes=config["data"]["orientation"], labels=None), # different models have different expected orientations, denoted in the example configs
            Spacingd(keys=['image'], pixdim=(1.0, 1.0, 1.0), mode='bilinear'),
            CropForegroundd(keys=['image'], source_key='image'),
            RandSpatialCropd(keys=['image'], roi_size=config["data"]["train_patch_shape"], random_size=False),  
            SpatialPadd(keys=['image'], spatial_size=config["data"]["train_patch_shape"], mode='constant'),
            RandShiftIntensityd(keys=['image'], offsets=0.1, prob=config["training"].get("shift_intensity", 0.0)),
            RandScaleIntensityd(keys=['image'], factors=0.1, prob=config["training"].get("scale_intensity", 0.0)),
            # If some dataset items include 'seg', remove it so batches are consistent
            DeleteItemsd(keys=['seg']),
            ToTensord(keys=["image", "label", "event"], dtype=torch.float32, track_meta=False)
        ])

    elif train_roi_type == "seg_weighted":
        train_transform = Compose([
        LoadImaged(keys=['image', "seg"], allow_missing_keys=True),
        EnsureChannelFirstd(keys=["image", "seg"], allow_missing_keys=True),
        get_normalization_transform(config),
        Orientationd(keys=['image', "seg"], axcodes=config["data"]["orientation"], allow_missing_keys=True, labels=None),        
        Spacingd(keys=['image', "seg"], pixdim=(1.0, 1.0, 1.0), mode='bilinear', allow_missing_keys=True),
        CropForegroundd(keys=['image', "seg"], source_key='image', allow_missing_keys=True),        
        # comment out if non binary weighting
        Lambdad(
            keys=['seg'],
            func=lambda x: (x > 0).astype(np.float32),  # Any non-zero label becomes 1, we could change if certain tumor regions want more weighting
            allow_missing_keys=True
        ),  
        SmartWeightedCrop( # handles the case that a patient doesnt have a seg available. TODO: should we add the option to enforce segs?
            keys=['image'],
            seg_key='seg',
            spatial_size=config["data"]["train_patch_shape"]
        ),
        # Don't need seg anymore, can remove it
        DeleteItemsd(keys=['seg']),
        SpatialPadd(
            keys=['image'], 
            spatial_size=config["data"]["train_patch_shape"],
            mode='constant'
        ),
        # optionally add a bit of augmentation
        RandShiftIntensityd(keys=['image'], offsets=0.1, prob=config["training"].get("shift_intensity", 0.0)),
        RandScaleIntensityd(keys=['image'], factors=0.1, prob=config["training"].get("scale_intensity", 0.0)),
        ToTensord(keys=["image", "label", "event"], dtype=torch.float32, track_meta=False, allow_missing_keys=False)
    ])
    else:
        raise ValueError(f"Unsupported train_roi_type: {train_roi_type}. Supported types: random, seg_weighted")
    
    val_roi_type = config["data"]["val_roi_type"]
    if val_roi_type == "center_crop":      
        print("Using center cropping for validation")  
        val_transform = Compose([
                LoadImaged(keys=['image']),
                EnsureChannelFirstd(keys=["image"]),
                get_normalization_transform(config),
                Orientationd(keys=['image'], axcodes=config["data"]["orientation"], labels=None),
                Spacingd(keys=['image'], pixdim=(1.0, 1.0, 1.0), mode='bilinear'),
                CropForegroundd(keys=['image'], source_key='image', margin=1),
                CenterSpatialCropd(keys=['image'], roi_size=config["data"]["val_patch_shape"]), 
                SpatialPadd(keys=['image'], spatial_size=config["data"]["val_patch_shape"], mode='constant'), 
                DeleteItemsd(keys=['seg']),
                ToTensord(keys=["image", "label", "event"], dtype=torch.float32, track_meta=False, allow_missing_keys=False)
            ])
        
    # could add seg_weighted to val transformations but not deterministic.

    elif val_roi_type == "tumor_centered":
        val_transform = Compose([
        LoadImaged(keys=['image', "seg"], allow_missing_keys=True),
        EnsureChannelFirstd(keys=["image", "seg"], allow_missing_keys=True),
        get_normalization_transform(config),
        Orientationd(keys=['image', "seg"], axcodes=config["data"]["orientation"], allow_missing_keys=True, labels=None),        
        Spacingd(keys=['image', "seg"], pixdim=(1.0, 1.0, 1.0), mode='bilinear', allow_missing_keys=True),
        CropForegroundd(keys=['image', "seg"], source_key='image', allow_missing_keys=True),       
        Lambdad(
            keys=['seg'],
            func=lambda x: (x > 0).astype(np.float32),
            allow_missing_keys=True
        ),  
        TumorCenterCrop(
            keys=['image'],
            seg_key='seg',
            spatial_size=config["data"]["val_patch_shape"]
        ),
        DeleteItemsd(keys=['seg']),
        SpatialPadd(
            keys=['image'], 
            spatial_size=config["data"]["val_patch_shape"],
            mode='constant'
        ),
        ToTensord(keys=["image", "label", "event"], dtype=torch.float32, track_meta=False, allow_missing_keys=False)
        ])

    else:
        raise ValueError(f"Unsupported val_roi_type: {val_roi_type}. Supported types: center_crop, tumor_centered")
        
    return train_transform, val_transform


"""
Some other augmentations to consider
RandGaussianSmoothd(keys=["image1", "image2", "image3", "image4"], prob=0.2),
RandGaussianNoised(keys=["image1", "image2", "image3", "image4"], prob=0.2, std=0.05),
RandAdjustContrastd(keys=["image1", "image2", "image3", "image4"], prob=0.2, gamma=(0.7, 1.3)),
"""