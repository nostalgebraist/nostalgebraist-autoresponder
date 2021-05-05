import subprocess

import torch
from selector_model.selector_nn_neo import get_child_module_by_names

from stable_library_code.transformers.gpt_neo.modeling_gpt_neo import (
    GPTNeoForCausalLM,
    GPTNeoModel,
)
from transformers.models.gpt_neo.configuration_gpt_neo import GPTNeoConfig

GPTNeoModel.init_weights = lambda *args, **kwargs: None
GPTNeoForCausalLM.init_weights = lambda *args, **kwargs: None


def show_gpu():
    print(subprocess.check_output("nvidia-smi").decode())


def ultra_defensive_load(config_path, model_path, verbose=True):
    def vshow_gpu():
        if verbose:
            show_gpu()

    vshow_gpu()

    # gpu memory
    state_dict = torch.load(
        model_path,
        map_location=torch.device("cuda:0"),
    )

    vshow_gpu()

    # uses lazy init, no memory
    model = GPTNeoForCausalLM(config=GPTNeoConfig.from_pretrained(config_path))

    vshow_gpu()

    # START gpu --> cpu --> gpu handoff, one leaf module at a time
    handled = set()

    for name in dict(model.named_parameters()).keys():
        prefix = name.rpartition(".")[0]
        mod = get_child_module_by_names(model, prefix.split("."))

        if prefix in handled:
            continue

        if verbose:
            print((name, prefix, mod))

        mk, uk, er = [], [], []
        mod._load_from_state_dict(
            state_dict,
            prefix=prefix + ".",
            local_metadata={},
            strict=True,
            missing_keys=mk,
            unexpected_keys=uk,
            error_msgs=er,
        )
        if verbose:
            print((mk, uk, er))
        mod.cuda()
        sdks = [k for k in state_dict if k.startswith(prefix)]
        for k in sdks:
            del state_dict[k]
        handled.add(prefix)

    # END gpu --> cpu --> gpu handoff, one leaf module at a time

    vshow_gpu()

    model.tie_weights()

    vshow_gpu()

    # does the buffers
    model = model.cuda()

    vshow_gpu()

    return model
