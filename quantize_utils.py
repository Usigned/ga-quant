import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import copy
import torchvision
from pytorch_ssim import ssim

from energy import energy_linear, energy_conv, network_energy

# uniform-quantizer
class QModule(nn.Module):
    def __init__(self, w_bit=-1, a_bit=-1, half_wave=True):
        '''
        half wave True: means usigned range, False: symmetric range

        '''
        super(QModule, self).__init__()

        if half_wave:
            self._a_bit = a_bit
        else:
            self._a_bit = a_bit - 1
        self._w_bit = w_bit
        self._b_bit = 32
        self._half_wave = half_wave

        self.init_range = 6.
        self.activation_range = nn.Parameter(torch.Tensor([self.init_range]))
        self.weight_range = nn.Parameter(torch.Tensor([-1.0]), requires_grad=False)

        self._quantized = True
        self._tanh_weight = False
        self._fix_weight = False
        self._trainable_activation_range = True
        self._calibrate = False

        self.eeval = False


    @property
    def w_bit(self):
        return self._w_bit

    @w_bit.setter
    def w_bit(self, w_bit):
        self._w_bit = w_bit

    @property
    def a_bit(self):
        if self._half_wave:
            return self._a_bit
        else:
            return self._a_bit + 1

    @a_bit.setter
    def a_bit(self, a_bit):
        if self._half_wave:
            self._a_bit = a_bit
        else:
            self._a_bit = a_bit - 1

    @property
    def b_bit(self):
        return self._b_bit

    @property
    def half_wave(self):
        return self._half_wave

    @property
    def quantized(self):
        return self._quantized

    @property
    def tanh_weight(self):
        return self._tanh_weight

    def set_quantize(self, quantized):
        self._quantized = quantized

    def set_tanh_weight(self, tanh_weight):
        self._tanh_weight = tanh_weight
        if self._tanh_weight:
            self.weight_range.data[0] = 1.0

    def set_fix_weight(self, fix_weight):
        self._fix_weight = fix_weight

    def set_activation_range(self, activation_range):
        self.activation_range.data[0] = activation_range

    def set_weight_range(self, weight_range):
        self.weight_range.data[0] = weight_range

    def set_trainable_activation_range(self, trainable_activation_range=True):
        self._trainable_activation_range = trainable_activation_range
        self.activation_range.requires_grad_(trainable_activation_range)

    def set_calibrate(self, calibrate=True):
        self._calibrate = calibrate

    def set_tanh(self, tanh=True):
        self._tanh_weight = tanh

    def _compute_threshold(self, data, bitwidth):
        mn = 0
        mx = np.abs(data).max()
        if np.isclose(mx, 0.0):
            return 0.0
        hist, bin_edges = np.histogram(np.abs(data), bins='sqrt', range=(mn, mx), density=True)
        hist = hist / np.sum(hist)
        cumsum = np.cumsum(hist)
        n = pow(2, int(bitwidth) - 1)
        threshold = []
        scaling_factor = []
        d = []
        if n + 1 > len(bin_edges) - 1:
            th_layer_out = bin_edges[-1]
            # sf_layer_out = th_layer_out / (pow(2, bitwidth - 1) - 1)
            return float(th_layer_out)
        for i in range(n + 1, len(bin_edges), 1):
            threshold_tmp = (i + 0.5) * (bin_edges[1] - bin_edges[0])
            threshold = np.concatenate((threshold, [threshold_tmp]))
            scaling_factor_tmp = threshold_tmp / (pow(2, bitwidth - 1) - 1)
            scaling_factor = np.concatenate((scaling_factor, [scaling_factor_tmp]))
            p = np.copy(cumsum)
            p[(i - 1):] = 1
            x = np.linspace(0.0, 1.0, n)
            xp = np.linspace(0.0, 1.0, i)
            fp = p[:i]
            p_interp = np.interp(x, xp, fp)
            x = np.linspace(0.0, 1.0, i)
            xp = np.linspace(0.0, 1.0, n)
            fp = p_interp
            q_interp = np.interp(x, xp, fp)
            q = np.copy(p)
            q[:i] = q_interp
            d_tmp = np.sum((cumsum - q) * np.log2(cumsum / q))  # Kullback-Leibler-J
            d = np.concatenate((d, [d_tmp]))

        th_layer_out = threshold[np.argmin(d)]
        # sf_layer_out = scaling_factor[np.argmin(d)]
        threshold = float(th_layer_out)
        return threshold

    def _quantize_activation(self, inputs):
        '''
        default: use init range 6.
        if _calibrate: use calibration set range
        if _trainable_activation_range: then trainable_activation_range

        self._quantized is set to True on init func
        默认使用_trainable_activation_range
        '''
        if self._quantized and self._a_bit > 0:
            if self._calibrate:
                if self._a_bit < 5:
                    threshold = self._compute_threshold(inputs.data.cpu().numpy(), self._a_bit)
                    estimate_activation_range = min(min(self.init_range, inputs.abs().max()), threshold)
                else:
                    estimate_activation_range = min(self.init_range, inputs.abs().max())
                # print('range:', estimate_activation_range, '  shape:', inputs.shape, '  inp_abs_max:', inputs.abs().max())
                self.activation_range.data = torch.tensor([estimate_activation_range], device=inputs.device)
                return inputs

            if self._trainable_activation_range:
                if self._half_wave:
                    ori_x = 0.5 * (inputs.abs() - (inputs - self.activation_range).abs() + self.activation_range)
                else:
                    ori_x = 0.5 * ((-inputs - self.activation_range).abs() - (inputs - self.activation_range).abs())
            else:
                if self._half_wave:
                    ori_x = inputs.clamp(0.0, self.activation_range)
                else:
                    ori_x = inputs.clamp(-self.activation_range, self.activation_range)

            scaling_factor = self.activation_range / (2. ** self._a_bit - 1.)
            x = ori_x.detach().clone()
            x.div_(scaling_factor).round_().mul_(scaling_factor)

            # STE
            # x = ori_x + x.detach() - ori_x.detach()
            return STE.apply(ori_x, x)
        else:
            return inputs

    def _quantize_weight(self, weight):
        '''
        _tanh_weight: ???
        _quantized:
            default: will reset weight_range
            if _calibrate: use calibrate range 

        self._quantized is set to True on init func
        '''
        if self._tanh_weight:
            weight = weight.tanh()
            weight = weight / weight.abs().max()

        if self._quantized and self._w_bit > 0:
            threshold = self.weight_range.item()
            if threshold <= 0:
                threshold = weight.abs().max().item()
                self.weight_range.data[0] = threshold

            if self._calibrate:
                if self._w_bit < 5:
                    threshold = self._compute_threshold(weight.data.cpu().numpy(), self._w_bit)
                else:
                    threshold = weight.abs().max()
                self.weight_range.data[0] = threshold
                return weight

            ori_w = weight

            scaling_factor = threshold / (pow(2., self._w_bit - 1) - 1.)
            w = ori_w.clamp(-threshold, threshold)
            # w[w.abs() > threshold - threshold / 64.] = 0.
            w.div_(scaling_factor).round_().mul_(scaling_factor)

            # STE
            if self._fix_weight:
                # w = w.detach()
                return w.detach()
            else:
                # w = ori_w + w.detach() - ori_w.detach()
                return STE.apply(ori_w, w)
        else:
            return weight

    def _quantize_bias(self, bias):
        '''
        by defualt: this method will not be trigered
        '''
        if bias is not None and self._quantized and self._b_bit > 0:
            if self._calibrate:
                return bias
            ori_b = bias
            threshold = ori_b.data.max() + 0.00001
            scaling_factor = threshold / (pow(2., self._b_bit - 1) - 1.)
            b = torch.clamp(ori_b.data, -threshold, threshold)
            b.div_(scaling_factor).round_().mul_(scaling_factor)
            # STE
            if self._fix_weight:
                return b.detach()
            else:
                # b = ori_b + b.detach() - ori_b.detach()
                return STE.apply(ori_b, b)
        else:
            return bias

    def _quantize(self, inputs, weight, bias):
        inputs = self._quantize_activation(inputs=inputs)
        weight = self._quantize_weight(weight=weight)
        # bias = self._quantize_bias(bias=bias)
        return inputs, weight, bias

    def forward(self, *inputs):
        raise NotImplementedError

    def extra_repr(self):
        return 'w_bit={}, a_bit={}, half_wave={}, tanh_weight={}'.format(
            self.w_bit if self.w_bit > 0 else -1, self.a_bit if self.a_bit > 0 else -1,
            self.half_wave, self._tanh_weight
        )


class STE(torch.autograd.Function):
    # for faster inference
    @staticmethod
    def forward(ctx, origin_inputs, wanted_inputs):
        return wanted_inputs.detach()

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs, None


class QConv2d(QModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 w_bit=-1, a_bit=-1, half_wave=True):
        super(QConv2d, self).__init__(w_bit=w_bit, a_bit=a_bit, half_wave=half_wave)
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters() 

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs):
        inputs, weight, bias = self._quantize(inputs=inputs, weight=self.weight, bias=self.bias)
        out =  F.conv2d(inputs, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        # 能耗评估
        if self.eeval:
            self.energy = energy_conv(inputs.shape, weight.shape, out.shape, self.w_bit)
            if hasattr(self, 'split') and self.split == True:
                self.energy += network_energy(out, 8)
        # 提取中间结果
        if hasattr(self, 'split') and self.split == True:
            self.tmp = out
            # print(self, end=' ')
        return out

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.w_bit > 0 or self.a_bit > 0:
            s += ', w_bit={}, a_bit={}'.format(self.w_bit, self.a_bit)
            s += ', half wave' if self.half_wave else ', full wave'
        if hasattr(self, 'split') and self.split == True:
            s += f', split={self.split}'
        return s.format(**self.__dict__)


class QLinear(QModule):
    def __init__(self, in_features, out_features, bias=True, w_bit=-1, a_bit=-1, half_wave=True):
        super(QLinear, self).__init__(w_bit=w_bit, a_bit=a_bit, half_wave=half_wave)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, inputs):
        inputs, weight, bias = self._quantize(inputs=inputs, weight=self.weight, bias=self.bias)
        out =  F.linear(inputs, weight=weight, bias=bias)
        # 能耗评估
        if self.eeval:
            self.energy = energy_linear(inputs.shape, out.shape, self.w_bit)
        # 提取中间结果
        if hasattr(self, 'split') and self.split == True:
            self.tmp = out
        return out

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)
        if self.w_bit > 0 or self.a_bit > 0:
            s += ', w_bit={w_bit}, a_bit={a_bit}'.format(w_bit=self.w_bit, a_bit=self.a_bit)
            s += ', half wave' if self.half_wave else ', full wave'
        if hasattr(self, 'split') and self.split == True:
            s += f', split={self.split}'
        return s


def set_fix_weight(model, fix_weight=True):
    if fix_weight:
        print('\n==> set weight fixed')
    for name, module in model.named_modules():
        if isinstance(module, QModule):
            module.set_fix_weight(fix_weight=fix_weight)


def build_index(qmodel, quantizable_type=[QConv2d, QLinear]):
    quantizable_idx = []
    for i, m in enumerate(qmodel.modules()):
        if type(m) in quantizable_type:
            quantizable_idx.append(i)
    return quantizable_idx


def uniform_quantize(qnet, w_bit=8, a_bit=8, skip_first=True):
    clear_qnet(qnet)
    idx = build_index(qnet)
    if skip_first:
        idx = idx[1: ]
    for i, layer in enumerate(qnet.modules()):
        if i not in idx:
            continue
        else:
            layer.w_bit = w_bit
            layer.a_bit = a_bit


def mixed_quantize_with_partition(qmodel, strategy, split, q_idx=None, a_bit=8):
    '''
    重置所有策略\n
    model.split = split\n
    首层不量化，activation统一8位(除了第一层)，分隔层被split=True\n
    split_idx in [-1, 0, 1, 2, ..., len(model)-1],  -1表示原始输入传云端，模型不量化，没有层被标记\n
    split = 0，由于跳过首层，模型也不量化，但第一层会被split标记
    '''
    clear_qnet(qmodel)
    qmodel.split = split
    q_idx = q_idx if q_idx is not None else build_index(qmodel)
    if split < 0:
        return
    assert len(q_idx[1: ]) == len(strategy), \
         'You should provide the same number of bit setting as layer list for weight quantization!'
    q_dict = {n: b for n, b in zip(q_idx[1: split+1], strategy)}

    for i, layer in enumerate(qmodel.modules()):
        if i == q_idx[split]:
            layer.split = True
        if i not in q_dict.keys():
            continue
        else:
            layer.w_bit = q_dict[i]
            layer.a_bit = a_bit


def mixed_quant_with_partition_bw(qmodel, strategy, split, block_type, a_bit=8):
    '''
    重置所有策略\n
    split in [0, lenQModel(qmodel, block_type)-1]
    '''
    clear_qnet(qmodel)
    q_idx = build_index(qmodel)
    if split < 0:
        return
    assert len(q_idx[1: ]) == len(strategy), \
         'You should provide the same number of bit setting as layer list for weight quantization!'
    q_dict = {n: b for n, b in zip(q_idx[1:], strategy)}
    next_block_idx = bw_split(qmodel, block_type, split)
    next_block_idx = next_block_idx if next_block_idx != -1 else 100000000000000
    qmodel.next_block_idx = next_block_idx

    count = 0
    for i, layer in enumerate(qmodel.modules()):
        if i in q_dict.keys() and i < next_block_idx:
            count += 1
            layer.w_bit = q_dict[i]
            layer.a_bit = a_bit
    print(f'{split+1} blocks, {count} layers')



def bw_split(qmodel, block_type, split):
    '''
    从第split个block分开，返回下一block的idx，如果没有则返回-1\n
    split in [0, ...]
    '''
    block_idx = build_index(qmodel, block_type)
    for i, m in enumerate(qmodel.modules()):
        if i == block_idx[split]:
            m.split = True
    return block_idx[split+1] if split + 1 < len(block_idx) else -1


def load_qnet(qmodel, path):
    ch = {n.replace('module.', ''): v for n, v in torch.load(path).items()}
    qmodel.load_state_dict(ch, strict=False)


def set_eeval(qnet, eeval, types=[QConv2d, QLinear]):
    for module in qnet.modules():
        if type(module) in types:
            module.eeval = eeval


def energy_eval(qnet, input_shape, device='cpu'):
    '''
    qnet should be mixed_quantized first \n
    input_shape (3, 32, 32) like 3-d \n
    will set_eeval(False) automatically \n
    split_idx in [-1, 0, 1, 2, ..., len(model)-1],  -1表示原始输入传云端，本地无能耗
    '''
    with torch.no_grad():
        split = qnet.split
        set_eeval(qnet, True)
        qnet.to(device)(torch.randn(1, *input_shape).to(device))
        e  = extract_energy(qnet)
        set_eeval(qnet, False)
        e_sum = 0.0
        edge_idx = build_index(qnet)[0: split+1]
        for k in e.keys():
            if k in edge_idx:
                e_sum += e[k]
        e_sum = e_sum if e_sum != 0 else network_energy(torch.randn(1, *input_shape).to(device), num_bit=32)
    return e_sum


def energy_eval_bw(qnet, input_shape, device='cpu'):
    with torch.no_grad():
        next_block_idx = qnet.next_block_idx
        set_eeval(qnet, True)
        qnet.to(device)(torch.randn(1, *input_shape).to(device))
        e  = extract_energy(qnet)
        set_eeval(qnet, False)
        e_sum = 0.0
        count = 0
        for k in e.keys():
            if k < next_block_idx:
                count += 1
                e_sum += e[k]
        e_sum = e_sum if e_sum != 0 else network_energy(torch.randn(1, *input_shape).to(device), num_bit=32)
        # print(f'{count} layers')
    return e_sum



def extract_energy(qnet, types=[QConv2d, QLinear]):
    total = {}
    for i, module in enumerate(qnet.modules()):
        if type(module) in types and hasattr(module, 'energy'):
            total.update({i: module.energy})
    return total


def load_qnet(qnet, path):
    ch = {n.replace('module.', ''): v for n, v in torch.load(path).items()}
    qnet.load_state_dict(ch, strict=False)
    return qnet


def extract_IR(qnet):
    '''
    提取IR，如果qnet.split_idx = -1即不分隔则返回None
    '''
    res = None
    for module in qnet.modules():
        if hasattr(module, 'tmp'):
            # print(module)
            res = module.tmp
    return res


def clear_qnet(qnet):
    '''
    重置量化策略 + 删除tmp
    '''
    if hasattr(qnet, 'split'):
        del qnet.split
    for module in qnet.modules():
        if hasattr(module, 'tmp'):
            del module.tmp
        if hasattr(module, 'split'):
            del module.split
        if type(module) in  [QConv2d, QLinear]:
            module._a_bit = -1
            module._w_bit = -1


def privacy_eval(qnet, sample, label, device='cpu', test_sample=False):
    with torch.no_grad():    
        # sample, label = iter(dataLoader).next()
        shape = sample.shape[-2: ]
        qnet, sample, label = qnet.to(device), sample.to(device), label.to(device)

        qnet.eval()
        output = qnet(sample)
        if test_sample:
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()
            print(f'top1: {correct[:, :1].sum() / sample.shape[0]}')

        ir = extract_IR(qnet)
        ir_images = [sample]
        if ir is not None:
            # print(ir.shape, end=' ')
            ir_images = ir2images(ir, shape=shape, device=device)
        return minMSE(ir_images, sample)
        # return minKL(ir_images, sample, qnet)


def maxSSIM(ir_images, input):
    ssims = []
    for ir_image in ir_images:
        ssims.append(ssim(ir_image, input))
    return max(ssims)

def minKL(ir_images, input, model):
    kls = []
    output = F.softmax(model(input))
    for ir_image in ir_images:
        output1 = F.softmax(model(ir_image))
        kls.append(kl_divergence(output, output1) / output.shape[0])
    return min(kls)


def kl_divergence(p, q):
    '''
    p, q首先需要进行softmax处理
    '''
    return (p * (torch.log(p) - torch.log(q))).sum()


def ir2images(ir, shape, device='cpu', normalized=False, mean=None, std=None):
    '''
    将IR映射为图片张量\n
    如：(64, 64, 34, 34)的IR 将被映射为64个 (64, 3, 32, 32)张量
    '''
    ir_images = []
    if ir.shape[1] == 3:
        return [ir]
    for i in range(ir.shape[1]):
        x = ir[:, i]
        x = torchvision.transforms.functional.resize(torch.stack([x, x, x], dim=1), shape)
        if normalized:
            x = torchvision.transforms.functional.normalize(x, mean=mean, std=std)
        ir_images.append(x.to(device))
    return ir_images


def lenQmodel(model, layer_type=[QConv2d, QLinear]):
    return len(build_index(model, layer_type))


def mse(img1, img2):
    return torch.mean(torch.square(img1 - img2))


def minMSE(ir_images, input, device='cuda'):
    mses = []
    for ir in ir_images:
        mses.append(mse(ir.to(device), input.to(device)))
    return min(mses)