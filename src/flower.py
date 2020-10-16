import os
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms

class Flower(Dataset):

    def __init__(self, path, shape=[224,224], pad=32, normalize=False, mode='val'):

        filenames = []
        labels = []

        folders = os.listdir(path)
        label = 0
        
        for f in folders:
            foldername = os.path.join(path,f)
            if not os.path.isdir(foldername):
                continue

            files = os.listdir(foldername)

            for name in files:
                fname = os.path.join(foldername, name)

                labels.append(label)
                filenames.append(fname)
            label+=1

        self.filenames = filenames
        self.labels = labels
        self.split = mode

        padded_size = [shape[0]+pad, shape[1]+pad]
        if normalize:
            self.preprocessor = {
                'train':transforms.Compose([
                    transforms.Resize(padded_size),
                    transforms.RandomCrop(shape),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ]),
                'test':transforms.Compose([
                    transforms.Resize(shape),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])
            }
        else:
            self.preprocessor = {
                'train':transforms.Compose([
                    transforms.Resize(padded_size),
                    transforms.RandomCrop(shape),
                    transforms.ToTensor(),
                ]),
                'test':transforms.Compose([
                    transforms.Resize(shape),
                    transforms.ToTensor(),
                ])
            }


    def preprocess(self, image):
        split = 'train' if self.split=='train' else 'test'
        transform = self.preprocessor[split]
        return transform(image)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        fname, label = self.filenames[i], self.labels[i]
        image = self.preprocess(Image.open(fname).convert('RGB'))
        return image, label
