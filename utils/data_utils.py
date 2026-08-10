import json
import math
import os
import logging

from monai.data import Dataset, CacheDataset, DataLoader

logger = logging.getLogger(__name__)


def dict_to_list(data):
    """MONAI Dataset requires a list indexed by integers.
    Convert {patient_id: record, ...} → [{patient_id: id, ...}, ...].
    Also coerces label/event to float/int so ToTensord can handle JSON files
    where these fields are stored as strings (e.g. "469" instead of 469).
    Keys unused by the transform pipeline ('dti', 'clinical') are dropped so
    MONAI's collate function never sees inconsistent keys across records
    (e.g. LOSO splits mix UPENN patients with DTI and non-UPENN without).
    """
    _DROP_KEYS = {"dti", "clinical"}
    if isinstance(data, dict):
        items = []
        for pid, record in data.items():
            item = {k: v for k, v in record.items() if k not in _DROP_KEYS}
            item["patient_id"] = pid
            item["label"] = float(item["label"])
            item["event"] = float(item["event"])
            items.append(item)
        # MONAI collate fails if keys are inconsistent across records in a batch.
        # If any record is missing 'seg' (e.g. BraTS records in LOSO splits),
        # drop it from all records so every batch sees the same key set.
        if items and not all("seg" in item for item in items):
            n_missing = sum(1 for item in items if "seg" not in item)
            logger.warning(
                f"'seg' missing in {n_missing}/{len(items)} records — "
                "dropping 'seg' from all records; seg-weighted transforms will be unavailable"
            )
            for item in items:
                item.pop("seg", None)
        return items
    return data


def drop_nan_labels(data, split_name):
    if data is None:
        return data
    if isinstance(data, dict):
        filtered = {k: v for k, v in data.items() if not math.isnan(float(v["label"]))}
        dropped = len(data) - len(filtered)
    else:
        filtered = [v for v in data if not math.isnan(float(v["label"]))]
        dropped = len(data) - len(filtered)
    if dropped:
        logger.warning(f"{split_name}: dropped {dropped} records with NaN label")
    return filtered


def load_split_data(json_path):
    """load a split JSON into train/validation/test record lists for MONAI"""
    with open(json_path, "r") as f:
        json_data = json.load(f)

    train_data = drop_nan_labels(dict_to_list(json_data.get("train", None)), "Training")
    val_data   = drop_nan_labels(dict_to_list(json_data.get("validation", None)), "Validation")
    test_data  = drop_nan_labels(dict_to_list(json_data.get("test", None)), "Testing")
    return train_data, val_data, test_data


def build_loaders(config, train_data, val_data, test_data, train_transforms, val_transforms, predict_only=False):
    """build train/val/test loaders plus a train loader with val transforms for final eval"""
    if predict_only:
        # no need to cache if we are only predicting
        train_cache_rate = 0.0
        val_cache_rate = 0.0
    else:
        train_cache_rate = config["data"].get("train_cache_rate", 0.0)
        val_cache_rate = config["data"].get("val_cache_rate", 0.0)

    # CacheDataset will cache up to the first Randomizable transformation, in our context this will mainly be the loading,
    # the orientation / spacing transforms, and normalizations
    train_ds = CacheDataset(data=train_data, transform=train_transforms, cache_rate=train_cache_rate, num_workers=4) if train_data else None
    val_ds   = CacheDataset(data=val_data,   transform=val_transforms, cache_rate=val_cache_rate, num_workers=4) if val_data else None
    # no need to cache the test set
    test_ds  = Dataset(data=test_data,  transform=val_transforms) if test_data else None
    # create a ds for the training set with validation transforms for final eval
    eval_train_ds = Dataset(data=train_data, transform=val_transforms) if train_data else None

    train_loader = DataLoader(train_ds, batch_size=config["data"]["batch_size"], shuffle=True, num_workers=config["data"]["workers"], pin_memory=True, persistent_workers=True) if train_ds else None
    val_loader   = DataLoader(val_ds,   batch_size=config["data"]["val_batch_size"], shuffle=False, num_workers=config["data"]["workers"], pin_memory=True, persistent_workers=True) if val_ds else None
    test_loader  = DataLoader(test_ds,  batch_size=config["data"]["val_batch_size"], shuffle=False, num_workers=config["data"]["workers"], pin_memory=True, persistent_workers=True) if test_ds else None
    eval_train_loader = DataLoader(eval_train_ds, batch_size=config["data"]["val_batch_size"], shuffle=False, num_workers=config["data"]["workers"], pin_memory=True, persistent_workers=True) if eval_train_ds else None

    return train_loader, val_loader, test_loader, eval_train_loader

