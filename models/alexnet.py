
import torch
import torch.nn as nn
from quantize_utils import QConv2d

class AlexNet(nn.Module):

  def __init__(self, classes=100, conv_layer=nn.Conv2d):
    super(AlexNet, self).__init__()
    self.features = nn.Sequential(
      conv_layer(3, 64, kernel_size=3, stride=2, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      conv_layer(64, 192, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      conv_layer(192, 384, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      conv_layer(384, 256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      conv_layer(256, 256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
    )
    self.classifier = nn.Sequential(
      nn.Dropout(),
      nn.Linear(256 * 1 * 1, 4096),
      nn.ReLU(inplace=True),
      nn.Dropout(),
      nn.Linear(4096, 4096),
      nn.ReLU(inplace=True),
      nn.Linear(4096, classes),
    )

  def forward(self, x):
    x = self.features(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x

def qalexnet(path=None, **kwargs):
    qnet = AlexNet(conv_layer=QConv2d, **kwargs)
    if path is not None:
        ch = {n.replace('module.', ''): v for n, v in torch.load(path).items()}
        qnet.load_state_dict(ch, strict=False)
    return qnet