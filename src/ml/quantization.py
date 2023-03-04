import torch
import os
import magma
import bitsandbytes as bnb

import ml.load_gptj
import config.autoresponder_config as arconfig

import transformers.models.gpt_neo.modeling_gpt_neo


def to_gpu(x, config):
    return x


transformers.models.gpt_neo.modeling_gpt_neo.to_gpu = to_gpu

ml.load_gptj.GPTNeoForCausalLM.init_weights = lambda self: None
transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoModel.init_weights = lambda self: None


class Linear8bitLtCompat(bnb.nn.Linear8bitLt):
    def __init__(self,
                 input_features, output_features, bias=True,
                 memory_efficient_backward=False, threshold=0.0, index=None):
        super(Linear8bitLtCompat, self).__init__(
            input_features, output_features, bias=bias,
            memory_efficient_backward=memory_efficient_backward,
            threshold=threshold,
            index=index,
            has_fp16_weights=False,
        )


def init_8bit(loading_code, **kwargs):
    def fn():
        ORIG_LINEAR = torch.nn.__dict__['Linear']

        torch.nn.__dict__['Linear'] = Linear8bitLtCompat
        try:
            result = loading_code(**kwargs)
        finally:
            torch.nn.__dict__['Linear'] = ORIG_LINEAR

        return result

    return fn


def load_gpt_j_8bit(ckpt_dir=arconfig.model_name):
    lm = init_8bit(ml.load_gptj.load_gpt_j_split_ckpt, ckpt_dir=ckpt_dir)()
    lm.cuda()
    return lm


def load_magma_8bit(path=arconfig.model_name, captioner_path):
    sd = ml.load_gptj.load_gpt_j_split_ckpt_state_dict(path)

    magma_config_path = os.path.join(captioner_path, 'config.yml')

    magma_wrapper = magma.Magma.from_split_checkpoint(
        path=captioner_path,
        config_path=magma_config_path,
        lm_path_or_state_dict=sd,
        gptj_init_fn=init_8bit(ml.load_gptj.quick_init_gptj),
        device='cpu',
        to_device=False,
    )

    return magma_wrapper

