"""Oxford-IIIT Pet test split (3,669 images) with its trimaps; pets have no ImageNet-1k class index here."""

from torchvision.datasets import OxfordIIITPet

from .masks import MaskDataset


class PetSegmentation(MaskDataset):
    def __init__(self, root, model, *, preprocessing=None):
        super().__init__(model, preprocessing=preprocessing)
        self.dataset = OxfordIIITPet(root, split="test", target_types="segmentation", download=False)

    def __len__(self):
        return len(self.dataset)

    def sample(self, index):
        image, trimap = self.dataset[index]  # trimap: 1 pet, 2 background, 3 boundary
        return image.convert("RGB"), -1, trimap
