import conf
import torch
from torch.nn.functional import softmax
import torch.optim as optim
import warnings
from progress.bar import Bar
import torchvision
from torchvision.transforms import transforms
from torch import log
import conf
import torchvision.transforms.functional as F
import torch.nn as nn

from quantize_utils import QConv2d, load_qnet



def has_childrean(module):
    try:
        next(module.children())
        return True
    except StopIteration:
        return False


def train(model, dataloader, optimizer, epoch=3, loss_fn=torch.nn.CrossEntropyLoss(), device='cpu', verbose=False):
    if device == 'cpu':
        warnings.warn('using cpu for training can be very slow', RuntimeWarning)
    model = model.to(device)
    with Bar('Training:', max=epoch*len(dataloader), suffix='%(percent)d%%') as bar:
        for e in range(epoch):
            for i, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                out = model(data)
                loss = loss_fn(out, target)
                loss.backward()
                optimizer.step()
                if verbose:
                    if i % 50 == 0:
                        print('Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                            e, i * len(data), len(dataloader.dataset), loss.item()
                        ))
                bar.next()


def test(model, dataloader, device='cpu'):
    model = model.to(device)
    model.eval()
    correct_1 = 0.0
    correct_5 = 0.0

    with torch.no_grad():
        for image, label in Bar('Testing').iter(dataloader):

            if device != 'cpu':
                image, label = image.to(device), label.to(device)

            output = model(image)
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            #compute top 5
            correct_5 += correct[:, :5].sum()

            #compute top1
            correct_1 += correct[:, :1].sum()

        top1, top5 = correct_1 / len(dataloader.dataset), correct_5 / len(dataloader.dataset)
        # print(top1, top5)
    return top1, top5


def finetune(qmodel, trainloader, epochs=1, device='cpu', need_test=True, testloader=None, verbose=False):
    '''
    finetune qmodel for #epochs(default 1) using SGD optimizer lr=0.0001, momentum=0.9
    '''
    optimizer = optim.SGD(qmodel.parameters(), lr=0.0001, momentum=0.9)
    train(qmodel, trainloader, optimizer, epoch=epochs, device=device, verbose=verbose)
    if need_test:
        if testloader is None:
            raise ValueError('need Test, but test dataloader is not given')
        top1, top5 = test(qmodel, testloader, device=device)
        return top1, top5


def cifar10DataLoader(root='/home/marcguo/lzq/work-dir/privacy-quantization/data/cifar-10-python', train=True, normalized=False, **kwargs):
    '''
    batch_size, num_workers, shuffle, pin_memory
    '''
    transform = transforms.Compose([
        transforms.RandomCrop(36, padding=4),
        transforms.CenterCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]) if normalized else transforms.ToTensor()

    dataLoader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.CIFAR10(
            root=root,
            download=False,
            transform=transform,
            train=train
        ),
        **kwargs
    )
    return dataLoader


def cifar100DataLoader(root='/home/marcguo/lzq/work-dir/privacy-quantization/data/', train=True, normalized=False, **kwargs):
    '''
    batch_size, num_workers, shuffle, pin_memory
    '''
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(conf.CIFAR100_TEST_MEAN, conf.CIFAR100_TEST_STD) if train else transforms.Normalize(conf.CIFAR100_TRAIN_MEAN, conf.CIFAR100_TRAIN_STD)
    ]) if normalized else transforms.ToTensor()

    
    dataLoader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.CIFAR100(
            root=root,
            download=False,
            transform=transform,
            train=train
        ),
        **kwargs
    )
    return dataLoader


def getCifar10Model(name, pretrained=True, normalized=False, quantized=True):
    '''
    alexnet   vgg11
    '''
    if name == 'alexnet':
        from models.alexnet import AlexNet
        model = AlexNet(classes=10)
        if pretrained:
            model.load_state_dict(torch.load(conf.alexnet_cifar10_no_nom) if normalized else conf.alexnet_cifar10_nom)
    elif name == 'vgg11':
        from models.vgg import vgg11_bn
        model = vgg11_bn(num_class=10, conv_layer=nn.Conv2d if not quantized else QConv2d)
        if pretrained:
            if not quantized:
                model.load_state_dict(torch.load(conf.vgg11_cifar10_no_nom))
            else:
                load_qnet(model, conf.vgg11_cifar10_no_nom)
    else:
        raise NotImplementedError(f'{name} is not supported yet')
    return model


def getCifar100Model(name, pretrained=True):
    '''
    qmodel\n
    normalized: mobilenetv2, shufflenetv2 resnet18 resnet50 vgg19
    '''
    if name == 'mobilenetv2':
        from models.mobilenetv2 import qmobilenetv2
        qnet = qmobilenetv2()
        if pretrained:
            load_qnet(qnet, conf.mobilenetv2_path)
    elif name == 'shufflenetv2':
        from models.shufflenetv2 import qshufflenetv2
        qnet = qshufflenetv2()
        if pretrained:
            load_qnet(qnet, conf.shufflenetv2_path)
    elif name == 'resnet18':
        from models.resnet import qresnet18
        qnet = qresnet18()
        if pretrained:
            load_qnet(qnet, conf.resnet18_path)
    elif name == 'resnet50':
        from models.resnet import qresnet50
        qnet = qresnet50()
        if pretrained:
            load_qnet(qnet, conf.resnet50_path)
    elif name =='vgg19':
        from models.vgg import qvgg19_bn
        qnet = qvgg19_bn()
        if pretrained:
            load_qnet(qnet, conf.vgg19_path)
    elif name == 'mobilenet':
        from models.mobilenet import qmobilenet
        qnet = qmobilenet()
        if pretrained:
            load_qnet(qnet, conf.mobilenet_path)
    else:
        raise NotImplementedError(f'{name} is not supported yet')     
    return qnet