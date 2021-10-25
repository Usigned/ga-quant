import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-net', required=True, help='net type', choices=['mobilenet', 'resnet18', 'resnet50'])
parser.add_argument('-Eb', help='energy bound', type=float, default=0.35)
parser.add_argument('-eb', help='loss bound', type=float, default=0.3)
parser.add_argument('-epoch', help='total GA iterations', type=int, default=1000)
parser.add_argument('-pop', help='population size', type=int, default=20)
parser.add_argument('-sample', help='sample size', type=int, default=10)
parser.add_argument('-mutate_prob_split', help='split mutate probability', type=float, default=0.5)
parser.add_argument('-mutate_prob_pi', help='pi mutate probability', type=float, default=0.3)
parser.add_argument('-finetune_e', help='finetune turns during the GA', type=int, default=1)
parser.add_argument('-lambda1', help='how strict the punishment is', type=float, default=0.1)
parser.add_argument('-lambda2', help='how strict the punishment is', type=float, default=0.1)
parser.add_argument('-b', help='batch size', type=int, default=256)
parser.add_argument('-device', help='device index', type=str, choices=['0', '1'], default='0, 1')
parser.add_argument('-resume', help='resume file path', type=str, default=None)
args = parser.parse_args()


import os
os.environ['CUDA_VISIBLE_DEVICES']= args.device

from es import GeneticAlgorithm, load_file
from models.resnet import *
from models.mobilenetv2 import *
from models.mobilenet import *
import conf


if args.net == 'resnet18':
    qmodel = qresnet18()
    block_type = [BasicBlock, BottleNeck]
    weight_path = conf.resnet18_path
elif args.net == 'resnet50':
    qmodel = qresnet50()
    block_type = [BasicBlock, BottleNeck]
    weight_path = conf.resnet50_path
elif args.net == 'mobilenet':
    qmodel = qmobilenet()
    block_type = [BasicConv2d, DepthSeperabelConv2d]
    weight_path = conf.mobilenet_path

# qmodel = qmobilenetv2()
# qmodel = qresnet18()
# qmodel = qresnet50()

ga = GeneticAlgorithm(
    arch=args.net,
    qmodel=qmodel,
    weight_path=weight_path,
    epoch=args.epoch, 
    pop_size=args.pop, 
    sample_size=args.sample, 
    mutate_prob_split=args.mutate_prob_split, 
    mutate_prob_pi=args.mutate_prob_pi,
    lambda1=args.lambda1,
    lambda2=args.lambda2,
    Eb=args.Eb,
    eb=args.eb,
    batch_size=args.b,
    block_type=block_type,
    finetune_e=args.finetune_e
)


pop = {}
if args.resume:
    load_file(pop, args.resume)

best = ga.run(verbose=True, init_pop=pop)