from .swin_transformer import build_swin
from .vision_transformer import build_vit
# from .mfm_raw import build_mfm

# from .mfm_local1 import build_mfm
from .mfm_local2 import build_mfm
# from .mfm_local3 import build_mfm

from .FourierTransformer import build_fourierVit

def build_model(config, is_pretrain=True):
    if is_pretrain:
        model = build_mfm(config)
    else:
        model_type = config.MODEL.TYPE
        if model_type == 'swin':
            model = build_swin(config)
        elif model_type == 'vit':
            model = build_vit(config)
        elif model_type == 'FourierVit':
            model = build_fourierVit(config)
        else:
            raise NotImplementedError(f"Unknown fine-tune model: {model_type}")

    return model
