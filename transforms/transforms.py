import monai.transforms as transforms
import numpy as np 
import torch

class SmartWeightedCrop(transforms.MapTransform):
    """
    Weighted crop if seg exists, otherwise random crop.
    Uses dictionary transforms to ensure same crop applied to all keys.
    """
    def __init__(self, keys, spatial_size, seg_key='seg'):
        super().__init__(keys)
        self.spatial_size = spatial_size
        self.seg_key = seg_key
        
        # Use the 'd' versions - they handle synchronization automatically
        self.weighted_crop = transforms.RandWeightedCropd(
            keys=keys,
            w_key=seg_key,
            spatial_size=spatial_size,
            num_samples=1,
        )
        self.random_crop = transforms.RandSpatialCropd(
            keys=keys, 
            roi_size=spatial_size,
            random_size=False,
            allow_missing_keys=True
        )
    
    def __call__(self, data):
        d = dict(data)
        
        has_seg = self.seg_key in d and d[self.seg_key] is not None
        
        if has_seg:
            d = self.weighted_crop(d)
        else:
            d = self.random_crop(d)
        
        return d

class TumorCenterCrop(transforms.MapTransform):
    """
    Deterministic crop centered on tumor center of mass.
    Falls back to brain center if no seg available.
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

            tumor_mask = seg > 0 # assuming 0 is background, can change if you want to center on a specific tumor region
            
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
        """Apply crop centered on given point"""
        img_shape = np.array(img.shape[1:])  # Spatial dimensions
        
        # Calculate start and end
        half_size = self.spatial_size // 2
        start = center - half_size
        end = start + self.spatial_size
        
        # Ensure within bounds
        start = np.maximum(0, start)
        end = np.minimum(img_shape, end)
        
        # Adjust to maintain spatial_size if possible
        for i in range(len(start)):
            if end[i] - start[i] < self.spatial_size[i]:
                if start[i] == 0:
                    end[i] = min(self.spatial_size[i], img_shape[i])
                elif end[i] == img_shape[i]:
                    start[i] = max(0, img_shape[i] - self.spatial_size[i])
        
        # Create slices
        slices = [slice(None)]  # Keep channel dimension
        slices.extend([slice(int(s), int(e)) for s, e in zip(start, end)])
        
        return img[tuple(slices)]
    

def get_crop_transform(config, keys, is_training=True):
    """
    Factory function to get the right crop transform based on config.
    """
    crop_type = config["training"]["train_crop_type"] if is_training else config["training"]["val_crop_type"]
    spatial_size = config["training"]["train_patch_shape"] if is_training else config["training"]["val_patch_shape"]
    
    if crop_type == "weighted":
        return SmartWeightedCrop(
            keys=keys,
            spatial_size=spatial_size,
            seg_key='seg',
        )
    elif crop_type == "tumor_center":
        return TumorCenterCrop(
            keys=keys,
            spatial_size=spatial_size,
            seg_key='seg',
            image_key='image',
        )
    elif crop_type == "center":
        return transforms.CenterSpatialCropd(
            keys=keys,
            roi_size=spatial_size,
        )
    elif crop_type == "random":
        return transforms.RandSpatialCropd(
            keys=keys,
            roi_size=spatial_size,
            random_size=False,
        )
    else:
        raise ValueError(f"Unknown crop type: {crop_type}")

def custom_transform(config):
    """
    Create custom transforms for training and validation
    """
    roi_type = config["training"].get("roi_type", "random")
    if roi_type == "random":
        print("Using random cropping for training and validation")
        
        train_transform = transforms.Compose([
            transforms.LoadImaged(keys=['image']),
            transforms.EnsureChannelFirstd(keys=["image"]),
            transforms.ScaleIntensityRangePercentilesd(keys=['image'], lower=5, upper=95, b_min=0.0, b_max=1.0, channel_wise=True),
            transforms.Orientationd(keys=['image'], axcodes='RAS'),
            transforms.Spacingd(keys=['image'], pixdim=(1.0, 1.0, 1.0), mode='bilinear'),
            # transforms.CropForegroundd(keys=['image'], source_key='image', margin=1),
            transforms.CropForegroundd(keys=['image'], source_key='image'),
            transforms.RandSpatialCropd(keys=['image'], roi_size=config["training"].get("train_patch_shape", 160), random_size=False),  
            transforms.SpatialPadd(keys=['image'], spatial_size=config["training"].get("train_patch_shape", 160), mode='constant'),
            # transforms.RandFlipd(keys=['image'], spatial_axis=(0, 1, 2), prob=0.5),
            transforms.RandShiftIntensityd(keys=['image'], offsets=0.1, prob=0.3),
            transforms.RandScaleIntensityd(keys=['image'], factors=0.1, prob=0.3),
            transforms.ToTensord(keys=["image", "label", "event"], dtype=torch.float32, track_meta=False)
        ])

    elif roi_type == "seg_weighted":
        train_transform = transforms.Compose([
        transforms.LoadImaged(keys=['image', "seg"], allow_missing_keys=True),
        transforms.EnsureChannelFirstd(keys=["image", "seg"], allow_missing_keys=True),
        transforms.ScaleIntensityRangePercentilesd(
            keys=['image'], lower=5, upper=95, 
            b_min=0.0, b_max=1.0, channel_wise=True
        ),
        transforms.Orientationd(keys=['image', "seg"], axcodes='RAS', allow_missing_keys=True),        
        transforms.Spacingd(keys=['image', "seg"], pixdim=(1.0, 1.0, 1.0), mode='bilinear', allow_missing_keys=True),
        transforms.CropForegroundd(keys=['image', "seg"], source_key='image', allow_missing_keys=True),        
        # not sure if we need to crop foreground if we are doing weighted crop later
        # comment out if you want non binary weighting
        transforms.Lambdad(
            keys=['seg'],
            func=lambda x: (x > 0).astype(np.float32),  # Any non-zero label becomes 1 
            allow_missing_keys=True
        ),  
        SmartWeightedCrop( # handles the case that a patient doesnt have a seg available
            keys=['image'],
            seg_key='seg',
            spatial_size=config["training"].get("train_patch_shape", 160),
        ),
        # Don't need seg anymore, can remove it
        transforms.DeleteItemsd(keys=['seg']),
        transforms.SpatialPadd(
            keys=['image'], 
            spatial_size=config["training"].get("train_patch_shape", 160), 
            mode='constant'
        ),
        transforms.RandShiftIntensityd(keys=['image'], offsets=0.1, prob=0.3),
        transforms.RandScaleIntensityd(keys=['image'], factors=0.1, prob=0.3),
        ToTensord(keys=["image", "label", "event"], dtype=torch.float32, track_meta=False, allow_missing_keys=False)
    ])
    else:
        raise ValueError(f"Unsupported roi_type: {roi_type}. Supported types: random, seg_weighted")
    
    val_roi_type = config["training"].get("val_roi_type", "random")
    
    if val_roi_type == "random":        
        val_transform = transforms.Compose([
                transforms.LoadImaged(keys=['image']),
                transforms.EnsureChannelFirstd(keys=["image"]),
                transforms.ScaleIntensityRangePercentilesd(keys=['image'], lower=5, upper=95, b_min=0.0, b_max=1.0, channel_wise=True),
                transforms.Orientationd(keys=['image'], axcodes='RAS'),
                transforms.Spacingd(keys=['image'], pixdim=(1.0, 1.0, 1.0), mode='bilinear'),
                transforms.CropForegroundd(keys=['image'], source_key='image', margin=1),
                transforms.RandSpatialCropd(keys=['image'], roi_size=config["training"].get("val_patch_shape", 160), random_size=False),  # Can remove
                transforms.SpatialPadd(keys=['image'], spatial_size=config["training"].get("val_patch_shape", 160), mode='constant'), # poss remove
                transforms.ToTensord(keys=["image", "label", "event"], dtype=torch.float32, track_meta=False)
            ])
    
    return train_transform, val_transform