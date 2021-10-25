import os

from torch.nn.modules import activation
os.environ['CUDA_VISIBLE_DEVICES']='1'
from torch.functional import split
from quantize_utils import build_index, energy_eval, load_qnet, mixed_quantize_with_partition, privacy_eval, set_fix_weight
from utils import finetune, getCifar100Model, getCifar10Model, cifar10DataLoader, cifar100DataLoader, test


def test_privacy(i):
    qmodel = getCifar10Model('vgg11', quantized=True)
    trainLoader = cifar10DataLoader(train=True, shuffle=True, batch_size=64)
    testLoader = cifar10DataLoader(train=False, batch_size=128, shuffle=False)

    mixed_quantize_with_partition(qmodel, [8] * (len(build_index(qmodel))-1), split=i)
    # energy_eval(qmodel, (3, 32, 32))
    return privacy_eval(qmodel, trainLoader, device='cuda')

def test_energy(i):
    qmodel = getCifar10Model('vgg11', quantized=True)
    mixed_quantize_with_partition(qmodel, [8] * (len(build_index(qmodel))-1), split=i)
    return energy_eval(qmodel, (3, 32, 32), device='cuda')


class Test:

    def __init__(self, normalized, dataset, batch_size=128, modelName=None, device='cuda') -> None:
        if dataset == 'cifar10':
            self.trainLoader = cifar10DataLoader(train=True, shuffle=False, batch_size=256)
            self.testLoader = cifar10DataLoader(train=False, batch_size=128, shuffle=False)
        else: 
            self.trainLoader = cifar100DataLoader(train=True, shuffle=False, batch_size=batch_size, normalized=normalized)
            self.testLoader = cifar100DataLoader(train=False, batch_size=batch_size, shuffle=False, normalized=normalized)
        self.sample, self.label = iter(self.trainLoader).next()
        self.dataset = dataset
        self.normalized = normalized
        self.device = device
        self.modelName = None
        self.model = None
        self.num_layers = None
        if modelName is not None:
            self.set_model(modelName, need_test=False)

    def set_model(self, modelName, need_test=False):
        self.modelName = modelName
        self.model = getCifar100Model(modelName) if self.dataset == 'cifar100' else getCifar10Model(modelName, normalized=self.normalized)
        self.num_layers = len(build_index(self.model))
        if need_test:
            test_top1, _ = test(self.model, self.testLoader, device=self.device)
            train_top1, _ = test(self.model, self.trainLoader, device=self.device)
            print(f'test top1 {test_top1}, train top1 {train_top1}')

    def test_privacy(self):
        for i in range(-1, self.num_layers):
            mixed_quantize_with_partition(self.model, strategy=[-1] * (self.num_layers-1), split=i, a_bit=-1)
            print(i+1, ',', privacy_eval(self.model, self.sample, self.label, self.device, test_sample=False).item())

    def test_energy(self):
        for i in range(-1, self.num_layers):
            mixed_quantize_with_partition(self.model, strategy=[8] * (self.num_layers-1), split=i)
            print(energy_eval(self.model, (3, 32, 32), device=self.device))

    def reload(self):
        self.set_model(self.modelName)

    def test_quantization(self, split):
        for i in [-1, 8, 6, 4, 2]:
            print('-----------------------------------------')
            print(f'{i} bit quantization')
            mixed_quantize_with_partition(self.model, [i] * (self.num_layers-1), split=split, a_bit=i)
            # print(finetune(qmodel=self.model, trainloader=self.trainLoader, epochs=1, device='cuda', testloader=self.trainLoader, verbose=False))
            # test_top1, _ = test(self.model, self.testLoader, device=self.device)
            # print(f'test top {test_top1}')
            privacy = privacy_eval(self.model, self.sample, self.label, device=self.device, test_sample=True)
            print(f'privacy: {privacy}')
            # self.reload()


if __name__ == '__main__':
    t = Test(normalized=True, dataset='cifar10', modelName='vgg11')
    t.test_privacy()
    # t.test_quantization(split=2)