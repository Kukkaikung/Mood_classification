import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from src.preprocess import apply_clahe


class FERDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.classes = sorted(os.listdir(self.root_dir))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = []
        for cls in self.classes:
            cls_path = os.path.join(self.root_dir, cls)
            for fname in os.listdir(cls_path):
                if fname.endswith('.jpg') or fname.endswith('.png'):
                    self.samples.append(
                        (os.path.join(cls_path, fname), self.class_to_idx[cls])
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(split='train'):
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.Lambda(apply_clahe),        # CLAHE added here
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.Lambda(apply_clahe),        # CLAHE added here too
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])


if __name__ == '__main__':
    dataset = FERDataset('data/raw', split='train', transform=get_transforms('train'))
    print(f'Total images : {len(dataset)}')
    print(f'Classes      : {dataset.classes}')
    print(f'Class map    : {dataset.class_to_idx}')

    image, label = dataset[0]
    print(f'Image shape  : {image.shape}')
    print(f'Label        : {label} ({dataset.classes[label]})')