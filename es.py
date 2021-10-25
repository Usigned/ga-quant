import json
import os
import os.path as osp
from quantize_utils import *
from utils import *
import random
import conf


class GeneticAlgorithm:

    def __init__(
        self, qmodel, weight_path, arch, block_type,
        epoch, pop_size, sample_size, mutate_prob_split, mutate_prob_pi, 
        r_w=[2, 4, 6, 8], r_a=8, 
        device='cuda', 
        input_size=(3, 32, 32),
        lambda1=0.8,
        lambda2=0.8,
        Eb = 0.7,
        eb = 0.3,
        log_dir='./log/',
        mem_dir='./mem/',
        batch_size=512, finetune_e=1) -> None:
        '''
        arch: 模型名 resnet18\n
        qmodel: 模型
        '''

        self.epoch = epoch
        self.pop_size = pop_size
        self.sample_size = sample_size
        self.r_w = r_w
        self.r_a = r_a
        self.mutate_prob_split = mutate_prob_split
        self.mutate_prob_pi = mutate_prob_pi
        self.qmodel = qmodel
        self.gene_length = lenQmodel(qmodel)
        self.pop = {}
        self.arch =arch
        self.finetue_e = finetune_e

        self.device = device
        self.input_size = input_size
        self.trainLoader = cifar100DataLoader(train=True, shuffle=True, normalized=True, batch_size=batch_size)
        self.testLoader = cifar100DataLoader(train=False, shuffle=False, normalized=True, batch_size=batch_size)
        self.sample, self.label = iter(self.trainLoader).next()
        
        self.weight_path = weight_path
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.Eb = Eb
        self.eb = eb
        self.log_file = osp.join(log_dir, self.arch, conf.TIME_NOW)
        os.makedirs(self.log_file)
        self.mem_dir = mem_dir
        # os.makedirs(osp.join(self.mem_dir, self.arxch))
        self.mem = []
        self.privacy_mem = {}
        self.energy_mem = {}
        self.accur_mem = {}

        self.block_type = block_type
        self.num_blocks = lenQmodel(self.qmodel, self.block_type)
        self.num_layers = lenQmodel(self.qmodel)

        load_file(self.privacy_mem, osp.join(mem_dir, self.arch, 'privacy.log'))
        load_file(self.energy_mem, osp.join(mem_dir, self.arch, 'energy.log'))
        load_file(self.accur_mem, osp.join(mem_dir, self.arch, 'accur.log'))

        load_qnet(self.qmodel, self.weight_path)
        self.top1, _ = test(self.qmodel, self.testLoader,device=self.device)
        self.top1 = self.top1.item()

    def start(self):
        self.pop = {}

    def init(self, pop):
        self.pop = pop

    def rand_init_pop(self):
        while len(self.pop) < self.pop_size:
            strategy = self.__random_init()
            if strategy in self.pop.keys():
                continue
            else:
                self.pop[strategy] = self.fitness_func(strategy)
    
    def __random_split(self):
        '''
        -1表示原始图片传输
        '''
        return random.randint(0, self.num_blocks-1) if self.num_blocks > 1 else 0

    def __random_bit(self):
        return random.sample(self.r_w, 1)[0]

    def __random_init(self):
        split = self.__random_split()
        quant_policy = tuple(self.__random_bit() for _ in range(self.gene_length - 1))
        return (split, quant_policy)

    def test_init(self):
        return self.__random_init()

    def random_sample(self):
        keys = random.sample(list(self.pop), self.sample_size)
        return {k: self.pop[k] for k in keys}

    def mutate(self, parent):
        split = parent[0]
        quant_policy = parent[1]
        new_split = split if random.random() > self.mutate_prob_split else self.__random_split()
        new_quant_policy = tuple([b if random.random() > self.mutate_prob_pi else self.__random_bit() for b in quant_policy])
        return (new_split, new_quant_policy)

    def selection(self, samples):
        '''
        best, worst
        '''
        # 大到小排序
        sorted_samples = sorted(samples.items(), key=lambda x: x[1], reverse=True)
        best, worst = sorted_samples[0][0], sorted_samples[-1][0]
        return best, worst

    def __add(self, strategy):
        if strategy in self.pop.keys():
            return
        else:
            self.pop[strategy] = self.fitness_func(strategy)

    def run(self, verbose=False, init_pop=None):
        # TODO: 添加文件读写 
        self.start()
        if init_pop:
            self.pop = init_pop
        self.rand_init_pop()
        for i in range(self.epoch):
            self.on_generation(idx=i)

            samples = self.random_sample()
            best, worst = self.selection(samples)
            if verbose:
                print(f'iteration: {i}\nbest: {best} : {self.pop[best]}\nworst: {worst} : {self.pop[worst]}\n')
            del self.pop[worst]
            while len(self.pop) < self.pop_size:
                children = self.mutate(best)
                self.__add(children)
        self.on_stop()
        return self.best()

    def on_generation(self, idx):
        '''
        每轮开始时动作，向文件中写入pop
        '''
        write_file(self.pop, osp.join(self.log_file, str(idx)))


    def on_stop(self):
        '''
        搜素结束时动作，将所有mem写入文件
        '''
        write_file(self.privacy_mem, osp.join(self.mem_dir, self.arch, 'privacy.log'))
        write_file(self.energy_mem, osp.join(self.mem_dir, self.arch, 'energy.log'))
        write_file(self.accur_mem, osp.join(self.mem_dir, self.arch, 'accur.log'))

    def best(self):
        '''
        strategy, accur, energy, privacy
        '''
        best_strategy, _ = self.selection(self.pop)
        return best_strategy, self.pop[best_strategy], self.accur_mem[best_strategy], self.energy_mem[best_strategy], self.privacy_mem[best_strategy]
    

    def fitness_func(self, strategy):
        split, pi = strategy
        load_qnet(self.qmodel, self.weight_path)
        mixed_quant_with_partition_bw(self.qmodel, strategy=pi, split=split, block_type=self.block_type, a_bit=self.r_a)

        if strategy not in self.mem:
            top1, _ = finetune(self.qmodel, self.trainLoader, epochs=self.finetue_e, device=self.device, testloader=self.testLoader)
            top1 = top1.item()
            privacy = privacy_eval(self.qmodel, self.sample, self.label, device=self.device).item()
            energy = energy_eval_bw(self.qmodel, self.input_size, device=self.device) / 1e9
            
            self.mem.append(strategy)
            self.accur_mem[strategy] = top1
            self.energy_mem[strategy] = energy
            self.privacy_mem[strategy] = privacy
        else:
            privacy, top1, energy = self.privacy_mem[strategy], self.accur_mem[strategy], self.energy_mem[strategy]

        fitness = privacy

        if energy > self.Eb:
            fitness *= self.lambda1 ** (energy / self.Eb - 1)
        if self.top1 - top1 > self.eb:
            fitness *= self.lambda2 ** ( (self.top1 - top1) / self.eb -1)

        return fitness


def load_file(dest, path):
    '''
    dest: dict k=(split, (*pi)), v=float\n
    path: path\n
    '''
    try:
        with open(path) as f:
            for line in f:
                (split, pi), value = json.loads(line)
                dest[(split, tuple(pi))] = value
        # print(f'{len(dest)} data loaded')
    except FileNotFoundError:
        # print('No file Found')
        pass

def write_file(src, path):
    '''
    dest: dict k=(split, (*pi)), v=float\n
    path: path\n
    '''
    with open(path, 'w') as f:
        for k, v in src.items():
            print(json.dumps((k, v)), file=f)
    # print(f'wirte {len(src)} data')