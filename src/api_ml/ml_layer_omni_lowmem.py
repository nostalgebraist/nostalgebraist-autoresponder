import io, sys, gc
import torch as th
import torch.nn as nn

import improved_diffusion.dist_util

_GLOBAL_FLAGS = {"DIFFUSION_DEVICE": 'cuda:0'}


def diffusion_device():
    return _GLOBAL_FLAGS['DIFFUSION_DEVICE']

improved_diffusion.dist_util.dev = diffusion_device

# load on gpu first to overusing cpu memory
import api_ml.ml_layer_diffusion

# offload diffusion to prep for text load
_GLOBAL_FLAGS['DIFFUSION_DEVICE'] = 'cpu'

del api_ml.ml_layer_diffusion.sampling_model_sres1
del api_ml.ml_layer_diffusion.sampling_model_sres1p5

gc.collect()
th.cuda.empty_cache()

for m in [
        api_ml.ml_layer_diffusion.sampling_model_sres2,
        api_ml.ml_layer_diffusion.sampling_model_sres3,
    ]:
    m.to(device=_GLOBAL_FLAGS['DIFFUSION_DEVICE'])

gc.collect()
th.cuda.empty_cache()

import api_ml.ml_layer_torch
# os.chdir("/nostalgebraist-autoresponder/")

_STDOUT_REF = sys.stdout


class AnnoyingCLIPLayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class FakeStream(io.IOBase):
    def write(self, *args, **kwargs): pass


def switch_to_diffusion():
    if hasattr(api_ml.ml_layer_torch, 'generator_model'):
        del api_ml.ml_layer_torch.generator_model.transformers_model
        del api_ml.ml_layer_torch.generator_model
        del api_ml.ml_layer_torch.magma_wrapper
        del api_ml.ml_layer_torch.selector_est
        del api_ml.ml_layer_torch.sentiment_est
        del api_ml.ml_layer_torch.autoreviewer_est
        gc.collect()
        th.cuda.empty_cache()

    _GLOBAL_FLAGS['DIFFUSION_DEVICE'] = 'cuda:0'

    if not hasattr(api_ml.ml_layer_diffusion, 'sampling_model_sres1'):
        # undo magma monkeypatch
        import clip.model
        clip.model.LayerNorm = AnnoyingCLIPLayerNorm

        api_ml.ml_layer_diffusion.sampling_model_sres1, api_ml.ml_layer_diffusion.sampling_model_sres1p5 = \
        api_ml.ml_layer_diffusion.load_part1_part1p5_curried()

        for m in [
                api_ml.ml_layer_diffusion.sampling_model_sres1,
                api_ml.ml_layer_diffusion.sampling_model_sres1p5,
                api_ml.ml_layer_diffusion.sampling_model_sres2,
                api_ml.ml_layer_diffusion.sampling_model_sres3,
            ]:
            m.to(device=_GLOBAL_FLAGS['DIFFUSION_DEVICE'])

    gc.collect()
    th.cuda.empty_cache()


def switch_to_text():
    _GLOBAL_FLAGS['DIFFUSION_DEVICE'] = 'cpu'

    if hasattr(api_ml.ml_layer_diffusion, 'sampling_model_sres1'):
        del api_ml.ml_layer_diffusion.sampling_model_sres1
        del api_ml.ml_layer_diffusion.sampling_model_sres1p5

    gc.collect()
    th.cuda.empty_cache()

    for m in [
            api_ml.ml_layer_diffusion.sampling_model_sres2,
            api_ml.ml_layer_diffusion.sampling_model_sres3,
        ]:
        m.to(device=_GLOBAL_FLAGS['DIFFUSION_DEVICE'])

    gc.collect()
    th.cuda.empty_cache()

    if not hasattr(api_ml.ml_layer_torch, 'generator_model'):
        # reapply magma monkeypatch
        import clip.model
        clip.model.LayerNorm = nn.LayerNorm

        api_ml.ml_layer_torch.generator_model, api_ml.ml_layer_torch.magma_wrapper = \
        api_ml.ml_layer_torch.load_generator_model_curried()

        api_ml.ml_layer_torch.selector_est = api_ml.ml_layer_torch.load_selector_curried()

        api_ml.ml_layer_torch.sentiment_est = api_ml.ml_layer_torch.load_sentiment_curried()

        api_ml.ml_layer_torch.autoreviewer_est = api_ml.ml_layer_torch.load_autoreviewer_curried()

        gc.collect()
        th.cuda.empty_cache()


def loop_poll_omni(period=1, text_polls_per_diffusion_poll=5, silence_text_stdout=False):
    fake = FakeStream()

    while True:
        try:
            if silence_text_stdout:
                sys.stdout = fake

            api_ml.ml_layer_torch.loop_poll(
                period=period,
                n_loops=text_polls_per_diffusion_poll,
                pre_hook=switch_to_text,
            )
            if silence_text_stdout:
                sys.stdout = _STDOUT_REF

            with th.cuda.amp.autocast():
                api_ml.ml_layer_diffusion.loop_poll(
                    period=period,
                    n_loops=1,
                    pre_hook=switch_to_diffusion,
                )
        except Exception as e:
            sys.stdout = _STDOUT_REF
            raise e
