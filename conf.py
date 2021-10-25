
from datetime import datetime
time_format = r'%Y_%m_%dT%H_%M_%S'

ori_top1 = 0.6887

ori_top5 = 0.9103

# accur 0.69
mobilenetv2_path = '/home/marcguo/lzq/work-dir/pytorch-cifar100/checkpoint/mobilenetv2/Monday_26_July_2021_19h_34m_17s/mobilenetv2-179-best.pth'
# mobilenetv2_path = './ckpt/mobilenetv2-179-best.pth'


# accur 0.7059
shufflenetv2_path = '/home/marcguo/lzq/work-dir/pytorch-cifar100/checkpoint/shufflenetv2/Tuesday_13_July_2021_16h_27m_00s/shufflenetv2-123-best.pth'
# shufflenetv2_path = './ckpt/shufflenetv2-123-best.pth'

# accur 0.7683
resnet18_path = '/home/marcguo/lzq/work-dir/pytorch-cifar100/checkpoint/resnet18/Wednesday_16_June_2021_14h_47m_09s/resnet18-190-best.pth'
# resnet18_path = './ckpt/resnet18-190-best.pth'

resnet50_path = '/home/marcguo/lzq/work-dir/pytorch-cifar100/checkpoint/resnet50/Friday_13_August_2021_14h_57m_54s/resnet50-192-best.pth'
# resnet50_path = './ckpt/resnet50-192-best.pth'

# alexnet_path = '/home/marcguo/lzq/work-dir/pytorch-alexnet-cifar100-master/checkpoints/cifar100_epoch_76.pth'

vgg19_path = '/home/marcguo/lzq/work-dir/pytorch-cifar100/checkpoint/vgg19/Tuesday_15_June_2021_17h_24m_51s/vgg19-189-best.pth'
# vgg19_path = './ckpt/vgg19-189-best.pth'

alexnet_cifar10_nom = '/home/marcguo/lzq/work-dir/privacy-quantization/model.pth'

alexnet_cifar10_no_nom = '/home/marcguo/lzq/work-dir/privacy-quantization/model_no_nom.pth'

vgg11_cifar10_no_nom = '/home/marcguo/lzq/work-dir/privacy-quantization/vgg11_no_nom.pth'

mobilenet_path = '/home/marcguo/lzq/work-dir/pytorch-cifar100/checkpoint/mobilenet/Sunday_10_October_2021_11h_35m_15s/mobilenet-183-best.pth'


TB_DIR = 'runs/'

TIME_NOW = datetime.today().strftime(time_format)


CIFAR100_TEST_MEAN = (0.5088964127604166, 0.48739301317401956, 0.44194221124387256)
CIFAR100_TEST_STD = (0.2682515741720801, 0.2573637364478126, 0.2770957707973042)

CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
