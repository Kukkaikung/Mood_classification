import sys
sys.path.append('.')
from src.dataset import FERDataset, get_transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

dataset_raw = FERDataset('data/raw', split='train', transform=None)
dataset = FERDataset('data/raw', split='train', transform=get_transforms('train'))

img_path, label = dataset_raw.samples[0]
emotion_name = dataset_raw.classes[label]

# before transform
before = Image.open(img_path).convert('RGB')

# after transform — convert tensor back to viewable image
tensor, _ = dataset[0]
# undo normalize to make it viewable
mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])
after = tensor.permute(1, 2, 0).numpy()  # [3,48,48] → [48,48,3]
after = (after * std + mean)             # undo normalize
after = np.clip(after, 0, 1)            # clip to valid range

# plot
fig, axes = plt.subplots(1, 2, figsize=(6, 3))

axes[0].imshow(before)
axes[0].set_title(f'before transform\nmode: RGB  size: {before.size}')
axes[0].axis('off')

axes[1].imshow(after)
axes[1].set_title(f'after transform\nshape: {tensor.shape}')
axes[1].axis('off')

fig.suptitle(f'emotion: {emotion_name}', fontsize=12)
plt.tight_layout()
plt.savefig('transform_check.png', dpi=150)
plt.show()
print("saved to transform_check.png")