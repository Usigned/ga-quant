import torch
import torch.nn as nn

_available_type = [nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, nn.Dropout]
_param_type = [nn.Conv2d, nn.Linear]

def MACs_energy(MACs, num_bits):
    return ((3.1 * num_bits) /32 + 0.1) * MACs

def mem_energy(mem, num_bits):
    return 2.5 * num_bits * mem

def has_childrean(module):
    try:
        next(module.children())
        return True
    except StopIteration:
        return False

def get_input_shape(dataloader):
    # TODO: need modification
    return (3, 32, 32)

def flatten(in_shape):
    result = 1
    for i in in_shape:
        result *= 1
    return result

def estimate_energy(model, input_shape, quant_policy):
    energy_layer = {}
    modules = [module for module in model.modules() if not has_childrean(module) and type(module) in _param_type]
    num_layers = len(modules)
    if num_layers != len(quant_policy):
        raise ValueError(f"#layers and #policy should be the same, but got {num_layers} and {len(quant_policy)}")
    for i, (module, bits) in enumerate(zip(modules, quant_policy)):
        if type(module) == nn.Conv2d:
            input_shape, e = conv_energy(module, input_shape, num_bits=bits)
            energy_layer[f'({i}) Conv2d [{bits}]'] = e
        elif type(module) == nn.Linear:
            input_shape, e = linear_energy(module, num_bits=bits)
            energy_layer[f'({i}) Linear [{bits}]'] = e
        elif type(module) == nn.ReLU:
            input_shape, e = relu_energy(input_shape)
        elif type(module) == maxpool_energy:
            input_shape, e = maxpool_energy(module, input_shape)
        else:
            pass
    return energy_layer


def energy_conv(input_size, weight_size, output_size, num_bits=8):
    '''
    mem = N^2 * I + p^2 * O * kernel_channels\n
    MAC = M^2 * O * p^2 * kernel_channels
    '''
    num_bits = num_bits if num_bits != -1 else 32
    
    mem = input_size[0] * input_size[1] * input_size[2] * input_size[3] + weight_size[0] * weight_size[1] * weight_size[2] * weight_size[3]
    MACs = output_size[2] * output_size[3] * weight_size[0] * weight_size[1] * weight_size[2] * weight_size[3]
    return mem_energy(mem, num_bits) + MACs_energy(MACs, num_bits)


def energy_linear(input_size, output_size, num_bits=8):
    num_bits = num_bits if num_bits != -1 else 32
    in_shape = input_size[1]
    out_shape = output_size[1]
    mem = in_shape + in_shape * out_shape
    MACs = in_shape * out_shape
    e = mem_energy(mem, num_bits=num_bits) + MACs_energy(MACs, num_bits)
    return e

def conv_energy(module, input_shape, num_bits=8):
    info = conv_module_info(module, input_shape)
    mem = info['in_size'][0] * info['in_size'][1] * info['in_channels'] + info['kernel_size'][0] * info['kernel_size'][1] * info['in_channels'] * info['out_channels']
    MACs =  info['out_size'][0] * info['out_size'][1] * info['in_channels'] * info['kernel_size'][0] * info['kernel_size'][1] * info['out_channels']
    e = mem_energy(mem, num_bits) + MACs_energy(MACs, num_bits)
    out_shape = info['out_channels'], info['out_size'][0], info['out_size'][1] 
    return out_shape, e

def linear_energy(module, num_bits=8):
    in_size, out_size = module.in_features, module.out_features
    mem = in_size + in_size * out_size
    MACs = in_size * out_size
    e = mem_energy(mem, num_bits) + MACs_energy(MACs, num_bits)
    out_shape = out_size
    return out_shape, e

def relu_energy(input_shape, num_bits=8):
    mem = input_shape if not isinstance(input_shape, tuple) else flatten(input_shape)
    MACs = mem
    e = mem_energy(mem, num_bits) + MACs_energy(MACs, num_bits)
    out_shape = input_shape
    return out_shape, e

def maxpool_energy(module, input_shape, num_bits=8):
    mem = 1
    for i in input_shape:
        mem *= i
    MACs = mem
    e = mem_energy(mem, num_bits) + MACs_energy(MACs, num_bits)
    out_shape = module(torch.rand(1, *input_shape)).squeeze().shape
    return out_shape, e

def convBNRelu_energy(module):
    # TODO: energy estimation for folding convBNRelu layer
    pass

def conv_module_info(module, input_shape):
    info = {}
    info['in_channels'], info['out_channels'], info['kernel_size'], info['stride'], info['padding'] = module.in_channels, module.out_channels, module.kernel_size, module.stride, module.stride
    info['in_size'] = input_shape[1], input_shape[2]
    out_shape = module(torch.rand(*input_shape).unsqueeze(0)).squeeze().shape
    info['out_size'] = out_shape[1], out_shape[2]
    return info

def compress_ratio(model, input_shape, policy, num_bits=8):
    '''
    base: num_bits uniq model
    '''
    base = sum(estimate_energy(model, input_shape, [num_bits] * len(policy)).values())
    e = sum(estimate_energy(model, input_shape, policy).values())
    return e / base


def network_energy(tensor, num_bit, verbose=False):
    '''
    4-d tensor, size * num_bit * 100pJ/b
    '''
    size = tensor.shape[0] * tensor.shape[1] * tensor.shape[2] * tensor.shape[3]
    if verbose:
        print(f'transmission size: {size} = {tensor.shape[0]} * {tensor.shape[1]} * {tensor.shape[2]} * {tensor.shape[3]}')
    return size * 100 * num_bit