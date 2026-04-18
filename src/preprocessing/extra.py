import albumentations as A

def get_extra_transforms() -> A.Compose:
    return A.Compose([
        # TODO: Add more complex transformations here
        # E.g., A.HorizontalFlip(p=0.5),
        #       A.RandomBrightnessContrast(p=0.2),
        #       A.Rotate(limit=30, p=0.5)
    ])
