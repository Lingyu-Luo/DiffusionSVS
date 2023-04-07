#########
# world
##########
import numpy as np
import torch
import parselmouth

gamma = 0
mcepInput = 3  # 0 for dB, 3 for magnitude
alpha = 0.45
en_floor = 10 ** (-80 / 20)
FFT_SIZE = 2048


f0_bin = 256
f0_max = 1100.0
f0_min = 50.0
f0_mel_min = 1127 * np.log(1 + f0_min / 700)
f0_mel_max = 1127 * np.log(1 + f0_max / 700)


def f0_to_coarse(f0):
    is_torch = isinstance(f0, torch.Tensor)
    f0_mel = 1127 * (1 + f0 / 700).log() if is_torch else 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1

    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
    f0_coarse = (f0_mel + 0.5).long() if is_torch else np.rint(f0_mel).astype(np.int)
    assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (f0_coarse.max(), f0_coarse.min())
    return f0_coarse


def norm_f0(f0, uv, hparams):
    is_torch = isinstance(f0, torch.Tensor)
    if hparams['pitch_norm'] == 'standard':
        f0 = (f0 - hparams['f0_mean']) / hparams['f0_std']
    if hparams['pitch_norm'] == 'log':
        f0 = torch.log2(f0) if is_torch else np.log2(f0)
    if uv is not None and hparams['use_uv']:
        f0[uv > 0] = 0
    return f0


def norm_interp_f0(f0, hparams):
    is_torch = isinstance(f0, torch.Tensor)
    if is_torch:
        device = f0.device
        f0 = f0.data.cpu().numpy()
    uv = f0 == 0
    f0 = norm_f0(f0, uv, hparams)
    if sum(uv) == len(f0):
        f0[uv] = 0
    elif sum(uv) > 0:
        f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
    uv = torch.FloatTensor(uv)
    f0 = torch.FloatTensor(f0)
    if is_torch:
        f0 = f0.to(device)
    return f0, uv


def denorm_f0(f0, uv, hparams, pitch_padding=None, min=None, max=None):
    if hparams['pitch_norm'] == 'standard':
        f0 = f0 * hparams['f0_std'] + hparams['f0_mean']
    if hparams['pitch_norm'] == 'log':
        f0 = 2 ** f0
    if min is not None:
        f0 = f0.clamp(min=min)
    if max is not None:
        f0 = f0.clamp(max=max)
    if uv is not None and hparams['use_uv']:
        f0[uv > 0] = 0
    if pitch_padding is not None:
        f0[pitch_padding] = 0
    return f0


'''
parselmouth
'''
def get_pitch_parselmouth(wav_data, mel, hparams):
    """

    :param wav_data: [T]
    :param mel: [T, 80]
    :param hparams:
    :return:
    """
    #time_step帧移，由公式计算： hop_size（帧移样本点数） = sample_rate*time_step/1000
    time_step = hparams['hop_size'] / hparams['audio_sample_rate'] * 1000
    f0_min = 40
    f0_max = 1600

    f0 = parselmouth.Sound(wav_data, hparams['audio_sample_rate']).to_pitch_ac(
        time_step=time_step / 1000, voicing_threshold=0.6,
        pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']
    pad_size = (int(len(wav_data) // hparams['hop_size']) - len(f0) + 1) // 2
    f0 = np.pad(f0, [[pad_size, len(mel) - len(f0) - pad_size]], mode='constant')
    pitch_coarse = f0_to_coarse(f0)
    return f0, pitch_coarse

def get_mel2ph(tg_fn, ph, mel, hparams):
    ph_list = ph.split(" ")
    with open(tg_fn, "r", encoding='utf-8') as f:
        tg = f.readlines()
    tg = remove_empty_lines(tg)
    tg = TextGrid(tg)
    tg = json.loads(tg.toJson())
    split = np.ones(len(ph_list) + 1, np.float) * -1
    tg_idx = 0
    ph_idx = 0
    tg_align = [x for x in tg['tiers'][-1]['items']]
    tg_align_ = []
    for x in tg_align:
        x['xmin'] = float(x['xmin'])
        x['xmax'] = float(x['xmax'])
        if x['text'] in ['sil', 'sp', '', 'SIL', 'PUNC']:
            x['text'] = ''
            if len(tg_align_) > 0 and tg_align_[-1]['text'] == '':
                tg_align_[-1]['xmax'] = x['xmax']
                continue
        tg_align_.append(x)
    tg_align = tg_align_
    tg_len = len([x for x in tg_align if x['text'] != ''])
    ph_len = len([x for x in ph_list if not is_sil_phoneme(x)])
    assert tg_len == ph_len, (tg_len, ph_len, tg_align, ph_list, tg_fn)
    while tg_idx < len(tg_align) or ph_idx < len(ph_list):
        if tg_idx == len(tg_align) and is_sil_phoneme(ph_list[ph_idx]):
            split[ph_idx] = 1e8
            ph_idx += 1
            continue
        x = tg_align[tg_idx]
        if x['text'] == '' and ph_idx == len(ph_list):
            tg_idx += 1
            continue
        assert ph_idx < len(ph_list), (tg_len, ph_len, tg_align, ph_list, tg_fn)
        ph = ph_list[ph_idx]
        if x['text'] == '' and not is_sil_phoneme(ph):
            assert False, (ph_list, tg_align)
        if x['text'] != '' and is_sil_phoneme(ph):
            ph_idx += 1
        else:
            assert (x['text'] == '' and is_sil_phoneme(ph)) \
                   or x['text'].lower() == ph.lower() \
                   or x['text'].lower() == 'sil', (x['text'], ph)
            split[ph_idx] = x['xmin']
            if ph_idx > 0 and split[ph_idx - 1] == -1 and is_sil_phoneme(ph_list[ph_idx - 1]):
                split[ph_idx - 1] = split[ph_idx]
            ph_idx += 1
            tg_idx += 1
    assert tg_idx == len(tg_align), (tg_idx, [x['text'] for x in tg_align])
    assert ph_idx >= len(ph_list) - 1, (ph_idx, ph_list, len(ph_list), [x['text'] for x in tg_align], tg_fn)
    mel2ph = np.zeros([mel.shape[0]], np.int)
    split[0] = 0
    split[-1] = 1e8
    for i in range(len(split) - 1):
        assert split[i] != -1 and split[i] <= split[i + 1], (split[:-1],)
    split = [int(s * hparams['audio_sample_rate'] / hparams['hop_size'] + 0.5) for s in split]
    for ph_idx in range(len(ph_list)):
        mel2ph[split[ph_idx]:split[ph_idx + 1]] = ph_idx + 1
    mel2ph_torch = torch.from_numpy(mel2ph)
    T_t = len(ph_list)
    dur = mel2ph_torch.new_zeros([T_t + 1]).scatter_add(0, mel2ph_torch, torch.ones_like(mel2ph_torch))
    dur = dur[1:].numpy()
    return mel2ph, dur
