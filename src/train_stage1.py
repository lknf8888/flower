from comet_ml import Experiment
import os

import torch
from torch.nn.functional import cross_entropy
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from flower import Flower

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

n_class = int(os.getenv('N_CLASS'))
epochs = int(os.getenv('N_EPOCH'))
batch_size = int(os.getenv('BATCH_SIZE'))
data_dir = os.getenv('DATA_FOLDER')
save_interval = int(os.getenv('SAVE_INTERVAL'))
save_path = os.getenv('SAVE_PATH')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
multi_gpus = device==torch.device('cuda') and torch.cuda.device_count() > 1


'''-----------------MODEL------------------------'''
enet_model = 'efficientnet-b0'
feat_size = 1280
use_bias = True

model = nn.Sequential(
    EfficientNet.from_pretrained(enet_model, include_top=False,),
    Reshape(-1,feat_size),
    nn.Linear(feat_size, n_class, bias=use_bias)
)


# save_path = 'save/sku10k-enet-b0-5-1000_freeze'
# checkpoint = save_path+'/best.pth'
# model.load_state_dict(torch.load(checkpoint, map_location=device) )

if multi_gpus:
    model = torch.nn.DataParallel(model, device_ids = range(torch.cuda.device_count()))
model.to(device)

'''-----------------MODEL------------------------'''

'''-------------------TRAINING----------------------------'''

dataset_path = '%s/flower_photos'%data_dir

trainset = Flower(path=dataset_path,mode='train')
train_loader = DataLoader(dataset=trainset,
                          batch_size=batch_size,
                          shuffle=True, num_workers=8)

for e in range(epochs):

    model.train()

    for i, (x,y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)

        logits = model(x_query)
        loss = cross_entropy(logits,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(e, loss.item())

    if e%save_interval == 0:
        torch.save(model.state_dict(), os.path.join(save_path, 'stage1_epoch-{}'.format(e) + '.pth'))
    if e==epochs-1:
        torch.save(model.state_dict(), os.path.join(save_path, 'stage1.pth'))
