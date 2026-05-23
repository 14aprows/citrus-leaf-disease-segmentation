from pathlib import Path
import numpy as np
import torch 
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

class CitrusLeafSegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, binary_mask=True, strict=True):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        self.binary_mask = binary_mask
        self.strict = strict

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory {self.image_dir} does not exist.")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Label directory {self.label_dir} does not exist.")
        
        self.images = sorted(path for path in self.image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        if not self.images:
            raise ValueError(f"No image files found in {self.image_dir}.")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        
        with Image.open(image_path) as image_file:
            image = np.array(image_file.convert("RGB"))
        
        label_path = self.label_dir / f"{image_path.stem}.txt"
        mask = self._load_polygon_mask(label_path, image.shape[:2])

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        
        if not torch.is_tensor(image):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        if not torch.is_tensor(mask):
            mask = torch.from_numpy(mask)
        
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        
        return image, mask.float()

    def _load_polygon_mask(self, label_path, image_shape):
        height, width = image_shape
        mask_image = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask_image)

        if not label_path.exists():
            if self.strict:
                raise FileNotFoundError(f"Label file {label_path} does not exist.")
            return np.array(mask_image, dtype=np.float32)
        
        for line_number, line in enumerate(label_path.read_text().splitlines(), start=1):
            line = line.strip()
            if not line:
                continue

            values = line.split()
            if len(values) < 7 or (len(values) - 1) % 2 != 0:
                raise ValueError(
                    f"Invalid line {line_number} in label file {label_path}."
                    "Expected class id followed by x y coordinate pairs"
                )
        
            class_id = int(float(values[0]))
            points = np.array(values[1:], dtype=np.float32).reshape(-1, 2)
            polygon = [
                (int(round(x * width)), int(round(y * height))) for x, y in points
            ]

            fill_value = 1 if self.binary_mask else class_id + 1
            draw.polygon(polygon, fill=fill_value)

        return np.array(mask_image, dtype=np.float32)