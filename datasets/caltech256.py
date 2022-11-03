import os.path as osp
from glob import glob
from PIL import Image
from torch.utils.data import Dataset

class Caltech256(Dataset):
    """
    NOTE 
    Due to Caltech256 include grayscale image (e.g. 001_0016.jpg, 006_0004.jpg, ...),
    torchvision.datasets.Caltech256 doesn't work.

    The following error occurs in Dataloader:
        RuntimeError: stack expects each tensor to be equal size, but got [3, 224, 224] at entry 0 and [1, 224, 224] at entry 11
    """

    def __init__(self, trans_X=None, trans_y=None, data_dir='data/caltech256'):

        self.paths = []
        self.labels = []
        self.trans_X = trans_X
        self.trans_y = trans_y

        paths = glob(osp.join(data_dir, '**', '*.jpg'), recursive=True)
        for path in paths:
            label = int(osp.basename(path).split('_')[0]) - 1   # label start at 0.
            self.paths.append(path)
            self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            label = self.labels[index]
            if image.mode == 'L':
                image = image.convert('RGB')    # grayscale to rgb
                
            if self.trans_X:
                image = self.trans_X(image)
            if self.trans_y:
                label = self.trans_y(label)

        return image, label


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torchvision.transforms import Compose, ToTensor, Resize, Normalize

    trans_X = Compose([Resize((224, 224)), ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    dataset = Caltech256(trans_X)
    dataloader = DataLoader(dataset)

    img, label = next(iter(dataloader))
    print(img.size())