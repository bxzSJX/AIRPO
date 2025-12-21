import torch
from torchvision import datasets, transforms


def fix_emnist_orientation(img: torch.Tensor):
    img = torch.rot90(img, k=1, dims=[1, 2])
    img = torch.flip(img, dims=[2])
    return img

def load_emnist(augment=False):

    if augment:
        #Training set: Add data augmentation to simulate the irregularities of handwriting.
        transform_train = transforms.Compose([
            transforms.RandomRotation(15, fill=0),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), fill=0),
            transforms.ToTensor(),
            transforms.Lambda(fix_emnist_orientation),
        ])
    else:
        # Test set: No enhancement
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(fix_emnist_orientation),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(fix_emnist_orientation),
    ])
    train_dataset = datasets.EMNIST(
        root="./data",
        split="balanced",
        train=True,
        download=True,
        transform=transform_train
    )

    test_dataset = datasets.EMNIST(
        root="./data",
        split="balanced",
        train=False,
        download=True,
        transform=transform_test
    )

    return train_dataset, test_dataset