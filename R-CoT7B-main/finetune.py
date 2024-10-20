# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.
import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
import numpy as np

import torch
import transformers
from accelerate.utils import DistributedType
from data_mix import Mix_dataset
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from peft import LoraConfig, get_peft_model
from transformers import Trainer, deepspeed
from transformers.trainer_pt_utils import LabelSmoother
from torch.utils.tensorboard import SummaryWriter

import torch_npu
from torch_npu.contrib import transfer_to_npu

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default='')


@dataclass
class DataArguments:
    data_path: str = field(
        default='data.txt', metadata={'help': 'Path to the training data.'})
    given_num: bool = False
    img_size: int = 224
    batch_size: int = 4


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default='adamw_torch')
    max_length: int = field(
        default=4096,
        metadata={
            'help':
            'Maximum sequence length. Sequences will be right padded (and possibly truncated).'
        },
    )
    use_lora: bool = False
    fix_vit: bool = True
    fix_sampler: bool = False
    label_names: List[str] = field(default_factory=lambda: ['samples'])
    save_total_limit: int = 4
    save_strategy: str = "steps"


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        'attention.wqkv',
        'attention.wo',
        'feed_forward.w1',
        'feed_forward.w2',
        'feed_forward.w3',
    ])
    lora_weight_path: str = ''
    lora_bias: str = 'none'


def maybe_zero_3(param):
    if hasattr(param, 'ds_id'):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == 'none':
        to_return = {k: t for k, t in named_params if 'lora_' in k}
    elif bias == 'all':
        to_return = {
            k: t
            for k, t in named_params if 'lora_' in k or 'bias' in k
        }
    elif bias == 'lora_only':
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if 'lora_' in k:
                to_return[k] = t
                bias_name = k.split('lora_')[0] + 'bias'
                lora_bias_names.add(bias_name)
            elif 'bias' in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str,
                                   bias='none'):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict(
        )
    else:
        # if trainer.args.use_lora:
        #     state_dict = get_peft_state_maybe_zero_3(
        #         trainer.model.named_parameters(), bias)
        # else:
        state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)


@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        instances = [instance['samples'] for instance in instances]
        text_input, data_type = tuple(
            [instance[key] for instance in instances]
            for key in ('text_input', 'data_type'))
        if 'image' not in instances[0]:
            text_input = [instance['text_input'][0] for instance in instances]
        batch = dict(
            text_input=text_input,
            data_type=data_type,
        )
        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            batch['image'] = images

        return dict(samples=batch)


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    rank0_print('Loading data...')
    if data_args.data_path.endswith('json'):
        train_json = json.load(open(data_args.data_path))
    elif data_args.data_path.endswith('txt'):
        train_json = {}
        with open(data_args.data_path) as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = line.split(' ')
            with open(line[0]) as f:
                temp = json.load(f)
            if data_args.given_num:  # 注意一下这个参数是什么作用
                assert len(line) == 2
                num = int(float(line[1]) * 1000)
                if len(temp) > num:
                    temp = random.sample(temp, num)
                else:
                    ex_temp = []
                    for i in range(num - len(temp)):
                        ex_temp.append(random.choice(temp))
                    temp.extend(ex_temp)
            else:
                if len(line) == 2:
                    ratio = float(line[1])
                    new_len = int(len(temp) * ratio)
                    if ratio < 1:
                        temp = random.sample(temp, new_len)
                    elif ratio > 1:
                        ex_temp = []
                        for i in range(new_len - len(temp)):  # 如果所需数据比所有数据多n，则循环n次，重复采样n个数据
                            ex_temp.append(random.choice(temp))
                        temp.extend(ex_temp)
            rank0_print(f'Load {len(temp)} samples from {line}')
            train_json[line[0]] = temp
    train_dataset = Mix_dataset(
        train_json,
        data_args.batch_size,
        img_size=data_args.img_size,
        local_rank=local_rank)
    print(str(len(train_dataset)) + 'samples is loaded')
    
    # test_path = '/root/paddlejob/workspace/env_run/dlluo/dle/InternLM-XComposer-main/finetune/duolun_json/geo3k_intern_test_choice_context.json'
    # with open(test_path) as f:
    #     eval_data = json.load(f)
    # image_paths = [item['image'][0] for item in eval_data]
    # texts = [item['conversations'][0]['value'] for item in eval_data]
    # labels = [item['conversations'][1]['value'] for item in eval_data]
    
    # test_json = {}
    # test_data = json.load(open(test_path))
    # test_json[test_path] = test_data
    # eval_dataset = Mix_dataset(
    #     test_json,
    #     data_args.batch_size,
    #     img_size=data_args.img_size,
    #     local_rank=local_rank)
    eval_dataset = None

    data_collator = DataCollatorForSupervisedDataset()
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.npu.is_available():
        torch.npu.manual_seed(seed)
        torch.npu.manual_seed_all(seed)  # 如果你使用多个 GPU

    # 确保所有的操作使用确定性的算法，这在有些情况下会降低性能
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train():
    global local_rank
    set_seed(77)

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments))
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    if getattr(training_args, 'deepspeed', None):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    local_rank = training_args.local_rank

    device_map = None

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False
    config.max_length = training_args.max_length

    # print(training_args)

    # Load model and tokenizer
    print(f'Load model from: {model_args.model_name_or_path}')
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        trust_remote_code=True,
    )

    if data_args.img_size != 336:
        model.vit.resize_pos()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        padding_side='right',
        use_fast=False,
        trust_remote_code=True,
    )
    model.tokenizer = tokenizer

    # import pdb
    # pdb.set_trace()
    if training_args.fix_vit:
        model.vit.requires_grad_(False)
    else:
        model.vit.requires_grad_(True)
        model.vit.vision_tower.vision_model.post_layernorm = torch.nn.Identity(
        )

    if training_args.fix_sampler:
        model.vision_proj.requires_grad_(False)
    else:
        model.vision_proj.requires_grad_(True)

    if training_args.use_lora:
        for name, param in model.model.named_parameters():
            param.requires_grad = False
        lora_config = LoraConfig(
            r=lora_args.lora_r,  # 低秩矩阵的秩。控制额外参数的数量。较低参数更少，减少了计算量和过拟合的风险。较高可以提高模型的表现能力，但同时增加了计算成本。
            lora_alpha=lora_args.lora_alpha,  # 控制 LoRA 适应中学习率的缩放
            target_modules=lora_args.lora_target_modules,  # 哪些模块或层将被 LoRA 所适配
            lora_dropout=lora_args.lora_dropout,  # LoRA 层中使用的 dropout 率
            bias=lora_args.lora_bias,  # 是否在 LoRA 适配中包括偏置项
            task_type='CAUSAL_LM',  # 指定了任务的类型。'CAUSAL_LM'是因果语言模型，也就是下一个词预测任。不同任务类型需要不同LoRA适配策略。
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    # for name, param in model.model.named_parameters():
    #         param.requires_grad = False

    # Load data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args)
    print(transformers.processing_utils.logging.is_progress_bar_enabled())
    transformers.processing_utils.logging.enable_progress_bar()

    # with open('lora_weights.txt', 'w') as f:
    #     for name, param in model.named_parameters():
    #         if 'lora' in name:  # 保存包含'LoRA'的权重参数
    #             f.write(f'{name}:\n')
    #             f.write(f'{param}\n\n')

    # for name, param in model.named_parameters():
    #     if 'lora' in name:  # 保存包含'LoRA'的权重参数
    #         print(f"Parameter: {name}, Requires gradient: {param.requires_grad}")
    # import pdb
    # pdb.set_trace()

    
    # training_args = training_args.set_save(strategy="steps", steps=500, total_limit = 4)
    print(training_args)
    # Start trainner
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module)

    trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(
        trainer=trainer,
        output_dir=training_args.output_dir,
        bias=lora_args.lora_bias)

    writer = SummaryWriter('runs/fashion_mnist_experiment_1')

if __name__ == '__main__':
    train()
