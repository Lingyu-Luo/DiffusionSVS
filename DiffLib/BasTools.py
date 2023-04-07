import librosa
import numpy as np
import os
import glob
import torch
from pycwt import wavelet
from scipy.interpolate import interp1d

def inverse_cwt_torch(Wavelet_lf0, scales):
    import torch
    b = ((torch.arange(0, len(scales)).float().to(Wavelet_lf0.device)[None, None, :] + 1 + 2.5) ** (-2.5))
    lf0_rec = Wavelet_lf0 * b
    lf0_rec_sum = lf0_rec.sum(-1)
    lf0_rec_sum = (lf0_rec_sum - lf0_rec_sum.mean(-1, keepdim=True)) / lf0_rec_sum.std(-1, keepdim=True)
    return lf0_rec_sum


def inverse_cwt(Wavelet_lf0, scales):
    b = ((np.arange(0, len(scales))[None, None, :] + 1 + 2.5) ** (-2.5))
    lf0_rec = Wavelet_lf0 * b
    lf0_rec_sum = lf0_rec.sum(-1)
    lf0_rec_sum = (lf0_rec_sum - lf0_rec_sum.mean(-1, keepdims=True)) / lf0_rec_sum.std(-1, keepdims=True)
    return lf0_rec_sum

def cwt2f0(cwt_spec, mean, std, cwt_scales):
    assert len(mean.shape) == 1 and len(std.shape) == 1 and len(cwt_spec.shape) == 3
    import torch
    if isinstance(cwt_spec, torch.Tensor):
        f0 = inverse_cwt_torch(cwt_spec, cwt_scales)
        f0 = f0 * std[:, None] + mean[:, None]
        f0 = f0.exp()  # [B, T]
    else:
        f0 = inverse_cwt(cwt_spec, cwt_scales)
        f0 = f0 * std[:, None] + mean[:, None]
        f0 = np.exp(f0)  # [B, T]
    return f0

#加载模型数据文件
def load_ckpt(cur_model, ckpt_base_dir, prefix_in_ckpt='model', force=True, strict=True):
    if os.path.isfile(ckpt_base_dir):
        base_dir = os.path.dirname(ckpt_base_dir)
        checkpoint_path = [ckpt_base_dir]
    else:
        base_dir = ckpt_base_dir
        checkpoint_path = sorted(glob.glob(f'{base_dir}/model_ckpt_steps_*.ckpt'), key=
        lambda x: int(re.findall(f'{base_dir}/model_ckpt_steps_(\d+).ckpt', x.replace('\\','/'))[0]))
    if len(checkpoint_path) > 0:
        checkpoint_path = checkpoint_path[-1]
        state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
        state_dict = {k[len(prefix_in_ckpt) + 1:]: v for k, v in state_dict.items()
                      if k.startswith(f'{prefix_in_ckpt}.')}
        if not strict:
            cur_model_state_dict = cur_model.state_dict()
            unmatched_keys = []
            for key, param in state_dict.items():
                if key in cur_model_state_dict:
                    new_param = cur_model_state_dict[key]
                    if new_param.shape != param.shape:
                        unmatched_keys.append(key)
                        print("| Unmatched keys: ", key, new_param.shape, param.shape)
            for key in unmatched_keys:
                del state_dict[key]
        cur_model.load_state_dict(state_dict, strict=strict)
        print(f"| load '{prefix_in_ckpt}' from '{checkpoint_path}'.")
    else:
        e_msg = f"| ckpt not found in {base_dir}."
        if force:
            assert False, e_msg
        else:
            print(e_msg)
