import cv2
import numpy as np
from PIL import Image


def apply_clahe(img_pil, clip_limit=2.0, tile_size=(4, 4)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to enhance low-light face images.

    Args:
        img_pil    : PIL Image (RGB)
        clip_limit : contrast enhancement strength
                     1.0 = subtle / 2.0 = balanced / 4.0 = aggressive
        tile_size  : grid size for local histogram equalization
                     (2,2) = fine detail / (4,4) = balanced / (8,8) = broad

    Returns:
        PIL Image (RGB) enhanced
    """
    # PIL → numpy
    img_np = np.array(img_pil.convert('RGB'))

    # RGB → LAB colorspace
    # L = brightness, A/B = color channels
    # we only touch L — color stays unchanged
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # apply CLAHE on L channel only
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_size
    )
    l_enhanced = clahe.apply(l)

    # merge enhanced L back with original A, B
    enhanced_lab = cv2.merge([l_enhanced, a, b])

    # LAB → RGB → PIL
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced_rgb)


def compare_clahe(img_pil, clip_limit=2.0, tile_size=(4, 4)):
    """
    Returns original and CLAHE-enhanced image for visual comparison.
    """
    original = img_pil.convert('RGB')
    enhanced = apply_clahe(img_pil, clip_limit, tile_size)
    return original, enhanced


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from src.dataset import FERDataset

    dataset = FERDataset('data/raw', split='train', transform=None)

    # test on 4 different images
    fig, axes = plt.subplots(4, 2, figsize=(6, 10))
    fig.suptitle('CLAHE Enhancement — before vs after', fontsize=12)

    for i in range(4):
        img_path, label = dataset.samples[i * 100]
        emotion = dataset.classes[label]

        img = Image.open(img_path)
        original, enhanced = compare_clahe(img)

        axes[i][0].imshow(original, cmap='gray' if img.mode == 'L' else None)
        axes[i][0].set_title(f'before — {emotion}', fontsize=9)
        axes[i][0].axis('off')

        axes[i][1].imshow(enhanced)
        axes[i][1].set_title(f'after CLAHE', fontsize=9)
        axes[i][1].axis('off')

    plt.tight_layout()
    plt.savefig('clahe_check.png', dpi=150)
    plt.show()
    print('saved to clahe_check.png')