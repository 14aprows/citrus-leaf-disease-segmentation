from torch.utils.data import DataLoader
from src.config import BATCH_SIZE, DATASET_DIR
from src.data.dataset import CitrusLeafSegmentationDataset

def create_dataset(split, dataset_dir = DATASET_DIR, transform=None, binary_mask=True, strict=True):
    split_dir = dataset_dir / split
    image_dir = split_dir / "images"
    label_dir = split_dir / "labels"

    return CitrusLeafSegmentationDataset(
        image_dir=image_dir, 
        label_dir=label_dir, 
        transform=transform, 
        binary_mask=binary_mask, 
        strict=strict
    )

def create_dataloader(split, dataset_dir = DATASET_DIR, transform=None, batch_size=BATCH_SIZE, shuffle=None, num_workers=0, pin_memory=False, binary_mask=True, strict=True):
    dataset = create_dataset(
        split=split, 
        dataset_dir=dataset_dir, 
        transform=transform, 
        binary_mask=binary_mask, 
        strict=strict
    )

    if shuffle is None:
        shuffle = split == "train"
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )

def create_dataloaders(dataset_dir = DATASET_DIR, train_transform=None, valid_transform=None, batch_size=BATCH_SIZE, num_workers=0, pin_memory=False):
    train_loader = create_dataloader(
        split="train", 
        dataset_dir=dataset_dir, 
        transform=train_transform, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )

    valid_loader = create_dataloader(
        split="valid", 
        dataset_dir=dataset_dir, 
        transform=valid_transform,
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )

    return train_loader, valid_loader