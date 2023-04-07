import os
from torch.autograd import Variable
import torch
import json
from os.path import join, exists
from glob import glob
from DiffLib.diffusion import GaussianDiffusion
from DiffLib.net import DiffNet
from DiffLib.nsf_hifigan import NsfHifiGAN
from DiffLib.phoneme import load_phoneme_list
from DiffLib.text_encoder import TokenTextEncoder
from DiffLib.slur_utils import merge_slurs
from DiffLib.audio import save_wav
from DiffLib.BasTools import load_ckpt
import numpy as np
import librosa
from DiffLib.tts_modules import LengthRegulator

PAD = "<pad>"
EOS = "<EOS>"
UNK = "<UNK>"
SEG = "|"
RESERVED_TOKENS = [PAD, EOS, UNK]
NUM_RESERVED_TOKENS = len(RESERVED_TOKENS)
PAD_ID = RESERVED_TOKENS.index(PAD)  # Normally 0
EOS_ID = RESERVED_TOKENS.index(EOS)  # Normally 1
UNK_ID = RESERVED_TOKENS.index(UNK)  # Normally 2
#参数定义
audio_num_mel_bins = 128
K_step = 1000
timesteps = 1000
diff_loss_type = 'l2'
spec_min = [-5]
spec_max = [0]
sample_rate = 44100
hop_size = 512
max_frames = 8000
pndm_speedup = 1

working_device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_parallel_device_ids = []
root_gpu = 0
if working_device == 'cuda':
    data_parallel_device_ids = [
        int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES","").split(",") if x != '']
    if len(data_parallel_device_ids) > 0:
        root_gpu = data_parallel_device_ids[0]

#预处理输入
def preprocess_phoneme_level_input(inp):
    ph_seq = inp['ph_seq']
    note_lst = inp['note_seq'].split()
    midi_dur_lst = inp['note_dur_seq'].split()
    is_slur = np.array(inp['is_slur_seq'].split(), 'float')
    ph_dur = None
    f0_timestep = float(inp['f0_timestep'])
    f0_seq = None
    if inp['f0_seq'] is not None:
        f0_seq = np.array(inp['f0_seq'].split(), 'float')
    if inp['ph_dur'] is not None:
        ph_dur = np.array(inp['ph_dur'].split(), 'float')
        print(len(note_lst), len(ph_seq.split()), len(midi_dur_lst), len(ph_dur))
        if len(note_lst) == len(ph_seq.split()) == len(midi_dur_lst) == len(ph_dur):
            print('Pass word-notes check.')
        else:
            print('The number of words does\'t match the number of notes\' windows. ',
                  'You should split the note(s) for each word by | mark.')
            return None
    else:
        print(len(note_lst), len(ph_seq.split()), len(midi_dur_lst))
        if len(note_lst) == len(ph_seq.split()) == len(midi_dur_lst):
            print('Pass word-notes check.')
        else:
            print('The number of words does\'t match the number of notes\' windows. ',
                  'You should split the note(s) for each word by | mark.')
            return None

    return ph_seq, note_lst, midi_dur_lst, is_slur, ph_dur, f0_timestep, f0_seq


def preprocess_input(inp, ph_encoder):
    """
    :param inp: {'text': str, 'item_name': (str, optional), 'spk_name': (str, optional)}
    :return:
    """

    item_name = inp.get('item_name', '<ITEM_NAME>')
    spk_name = inp.get('spk_name', 'opencpop')

    # single spk
    spk_id = 0

    # get ph seq, note lst, midi dur lst, is slur lst.
    ph_seq, note_lst, midi_dur_lst, is_slur, ph_dur, f0_timestep, f0_seq = preprocess_phoneme_level_input(inp)

    # convert note lst to midi id; convert note dur lst to midi duration
    midis = [librosa.note_to_midi(x.split("/")[0]) if x != 'rest' else 0
                 for x in note_lst]
    midi_dur_lst = [float(x) for x in midi_dur_lst]

    ph_token = ph_encoder.encode(ph_seq)
    item = {'item_name': item_name, 'text': inp['text'], 'ph': ph_seq, 'spk_id': spk_id,
            'ph_token': ph_token, 'pitch_midi': np.asarray(midis), 'midi_dur': np.asarray(midi_dur_lst),
            'is_slur': np.asarray(is_slur), 'ph_dur': None, 'f0_timestep': 0., 'f0_seq': None}
    item['ph_len'] = len(item['ph_token'])
    item['ph_dur'] = ph_dur
    item['f0_timestep'] = f0_timestep
    item['f0_seq'] = f0_seq

    return item

def input_to_batch(item):
    item_names = [item['item_name']]
    text = [item['text']]
    ph = [item['ph']]
    txt_tokens = torch.LongTensor(item['ph_token'])[None, :].to(working_device)
    txt_lengths = torch.LongTensor([txt_tokens.shape[1]]).to(working_device)
    spk_ids = torch.LongTensor(item['spk_id'])[None, :].to(working_device)

    pitch_midi = torch.LongTensor(item['pitch_midi'])[None, :max_frames].to(working_device)
    midi_dur = torch.FloatTensor(item['midi_dur'])[None, :max_frames].to(working_device)
    is_slur = torch.LongTensor(item['is_slur'])[None, :max_frames].to(working_device)
    mel2ph = None
    log2f0 = None
    if item['ph_dur'] is not None:
        #print('Using manual phone duration')
        ph_acc = np.around(
                np.add.accumulate(item['ph_dur']) * sample_rate / hop_size + 0.5).astype(
                'int')
        ph_dur = np.diff(ph_acc, prepend=0)
        ph_dur = torch.LongTensor(ph_dur)[None, :max_frames].to(working_device)
        lr = LengthRegulator()
        mel2ph = lr(ph_dur, txt_tokens == 0).detach()
    else:
        print('Using automatic phone duration')

    if item['f0_timestep'] > 0. and item['f0_seq'] is not None:
        #print('Using manual pitch curve')
        f0_timestep = item['f0_timestep']
        f0_seq = item['f0_seq']
        t_max = (len(f0_seq) - 1) * f0_timestep
        dt = hop_size / sample_rate
        f0_interp = np.interp(np.arange(0, t_max, dt), f0_timestep * np.arange(len(f0_seq)), f0_seq)
        log2f0 = torch.FloatTensor(np.log2(f0_interp))[None, :].to(working_device)
    else:
        print('Using automaic pitch curve')

    batch = {
        'item_name': item_names,
        'text': text,
        'ph': ph,
        'txt_tokens': txt_tokens,
        'txt_lengths': txt_lengths,
        'spk_ids': spk_ids,
        'pitch_midi': pitch_midi,
        'midi_dur': midi_dur,
        'is_slur': is_slur,
        'mel2ph': mel2ph,
        'log2f0': log2f0
    }
    return batch

def forward_model(model, inp, vocoder,return_mel=False):
    sample = input_to_batch(inp)
    txt_tokens = sample['txt_tokens']  # [B, T_t]
    spk_id = sample.get('spk_ids')
    with torch.no_grad():
        output = model(txt_tokens, spk_id=spk_id, ref_mels=None, infer=True,
                            pitch_midi=sample['pitch_midi'], midi_dur=sample['midi_dur'],
                            is_slur=sample['is_slur'], mel2ph=sample['mel2ph'], f0=sample['log2f0'])
        mel_out = output['mel_out']  # [B, T,80]
        f0_pred = output['f0_denorm']
        wav_out = vocoder.spec2wav_torch(mel_out, f0=f0_pred)
    wav_out = wav_out.cpu().numpy()
    return wav_out

def cross_fade(a: np.ndarray, b: np.ndarray, idx: int):
    result = np.zeros(idx + b.shape[0])
    fade_len = a.shape[0] - idx
    np.copyto(result[:idx], a[:idx])
    k = np.linspace(0, 1.0, num=fade_len, endpoint=True)
    result[idx: a.shape[0]] = (1 - k) * a[idx:] + k * b[: fade_len]
    np.copyto(result[a.shape[0]:], b[fade_len:])
    return result

#generate song wav from one *.ds file
def test_one_diff(label_path,diff_model,ph_encoder,vocoder):
    #读入输入文件
    with open(label_path, 'r', encoding='utf-8') as f:
        ds_items = json.load(f)
    #param_have_f0 = ds_items['f0_seq']
    #整合slur音
    result = np.zeros(0)
    current_length = 0
    for ds_item in ds_items:
        merge_slurs(ds_item)
        inp = preprocess_input(ds_item,ph_encoder)
        seg_audio = forward_model(diff_model,inp,vocoder)

        silent_length = round(ds_item.get('offset', 0) * sample_rate) - current_length
        if silent_length >= 0:
            result = np.append(result, np.zeros(silent_length))
            result = np.append(result, seg_audio)
        else:
            result = cross_fade(result, seg_audio, current_length + silent_length)
        current_length = current_length + silent_length + seg_audio.shape[0]
    return result


DATA_ROOT = "D:/DiffSong/"
MODEL_PATH = "D:/DiffSong/checkpoints/1226_Yujin_ds1000/model_ckpt_steps_182000.ckpt"
PHONEME_DICT_PATH = 'D:/DiffSong/checkpoints/1226_Yujin_ds1000/opencpop-strict.txt'
VOCODER_PATH = 'D:/DiffSong/checkpoints/nsf_hifigan/model'

#DATA_ROOT = "D:/DiffSong/"
#MODEL_PATH = "D:/DiffSong/checkpoints/Yujin/"
#PHONEME_DICT_PATH = 'D:/DiffSong/dictionaries/opencpop-strict.txt'
#VOCODER_PATH = 'D:/DiffSong/checkpoints/nsf_hifigan/model'
#data_path = 'D:/DiffSong/data/Yujin/binary'

# 读入音素字典
phone_list, pinyin2phs, phone_id, id_phoneme = load_phoneme_list(PHONEME_DICT_PATH)

ph_encoder = TokenTextEncoder(vocab_list=phone_list, replace_oov=',')

#准备模型
model = GaussianDiffusion(
            phone_encoder=ph_encoder,
            out_dims=audio_num_mel_bins, denoise_fn=DiffNet(audio_num_mel_bins),
            timesteps=timesteps,
            K_step=K_step,
            loss_type=diff_loss_type,
            spec_min=spec_min, spec_max=spec_max,
        )
model.eval()
#读入模型数据
load_ckpt(model, MODEL_PATH, 'model')
model.eval()
model.to(working_device)

#准备声码器
vocoder = NsfHifiGAN()
vocoder.model.eval()
vocoder.model.to(working_device)

torch.manual_seed(torch.seed() & 0xffff_ffff)
torch.cuda.manual_seed_all(torch.seed() & 0xffff_ffff)

#Reading ds files
test_label_paths = sorted(glob(join(DATA_ROOT, "samples/", "*.ds")))

#create generated file directory
save_dir = os.path.join("generated/synwav")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for label_path in test_label_paths:
    (path, filename) = os.path.split(label_path)
    fname = os.path.splitext(filename)
    if fname[0]!="妄语人间":
        continue
    waveform = test_one_diff(label_path,model,ph_encoder,vocoder)
    tmpfilename = join(save_dir, fname[0] + ".wav")
    print(f'| save audio: {tmpfilename}')
    save_wav(waveform, tmpfilename, sample_rate)
    '''
    starWavfile=starWav()
    starWavfile.audioFormat = WAVE_FORMAT_PCM
    starWavfile.numChannels = 1
    starWavfile.sampleRate = fs
    starWavfile.Tracks[0] = waveform
    starWavfile.bitsPerSample = 16
    tmpfilename = join(save_dir, filename+".wav")
    starWavfile.writePCM(tmpfilename)
    '''

















#load diff model
def load_ds_file(label_path):
    with open(label_path, mode='r', encoding='utf-8') as f:
        lines = [line.rstrip('\r\n') for line in f.readlines()]
    dsfile_items = dict()
    for line in lines:
        line = line.strip()
        if('"text":' in line ):
            line = line.replace('"text":','')
            dsfile_items['text'] = line.replace('"','').strip()
        elif ('"ph_seq":' in line):
            line = line.replace('"ph_seq":', '')
            dsfile_items['ph_seq'] = line.replace('"','').strip()
        elif ('"note_seq":' in line):
            line = line.replace('"note_seq":', '')
            dsfile_items['note_seq'] = line.replace('"','').strip()
        elif ('"note_dur_seq":' in line):
            line = line.replace('"note_dur_seq":', '')
            dsfile_items['note_dur_seq'] = line.replace('"','').strip()
        elif ('"is_slur_seq":' in line):
            line = line.replace('"is_slur_seq":', '')
            dsfile_items['is_slur_seq'] = line.replace('"','').strip()
        elif ('"ph_dur":' in line):
            line = line.replace('"ph_dur":', '')
            dsfile_items['ph_dur'] = line.replace('"','').strip()
        elif ('"f0_timestep":' in line):
            line = line.replace('"f0_timestep":', '')
            dsfile_items['f0_timestep'] = line.replace('"','').strip()
        elif ('"f0_seq":' in line):
            line = line.replace('"f0_seq":', '')
            dsfile_items['f0_seq'] = line.replace('"','').strip()
        elif ('"input_type":' in line):
            line = line.replace('"input_type":', '')
            dsfile_items['input_type'] = line.replace('"','').strip()
        elif ('"offset":' in line):
            line = line.replace('"offset":', '')
            dsfile_items['offset'] = line

    return dsfile_items