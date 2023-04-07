import os
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import pickle
from librosa import note_to_midi
from matplotlib import pyplot as plt
from DiffLib.nsf_hifigan import NsfHifiGAN
from DiffLib.phoneme import get_all_vowels
from DiffLib.text_encoder import TokenTextEncoder
from DiffLib.pitch_utils import get_pitch_parselmouth
import json
from config import hparams

#声码器
vocoder = None
vocoder = NsfHifiGAN()

#歌曲描述文件：
#wav文件名、歌曲文本、音素、音素时长
BASE_ITEM_ATTRIBUTES = ['txt', 'ph', 'wav_fn', 'tg_fn', 'spk_id']

class SongDesFile:
    def __init__(self, processed_data_dir=None, item_attributes=BASE_ITEM_ATTRIBUTES):
        if processed_data_dir is None:
            processed_data_dir = hparams['processed_data_dir']
        self.processed_data_dirs = processed_data_dir.split(",")
        self.binarization_args = hparams['binarization_args']
        self.pre_align_args = hparams['pre_align_args']

        self.items = {}
        # every item in self.items has some attributes
        self.item_attributes = item_attributes

        self.items = self.LoadSongDesFile(hparams['song_description_file'])
        self.item_names = sorted(list(self.items.keys()))

        if self.binarization_args['shuffle']:
            random.seed(1234)
            random.shuffle(self.item_names)

        item_names = deepcopy(self.item_names)
        self.test_item_names = [x for x in item_names if any([x.startswith(ts) for ts in hparams['test_prefixes']])]
        self.train_item_names = [x for x in item_names if x not in set(self.test_item_names)]
        self.valid_item_names = self.test_item_names
        #logging.info("train {}".format(len(train_item_names)))
        #logging.info("test {}".format(len(test_item_names)))

        # set default get_pitch algorithm
        self.get_pitch_algorithm = get_pitch_parselmouth

    # 读入歌声描述文件
    def LoadSongDesFile(self,FileName):
        raw_data_dir = hparams['raw_data_dir']
        utterance_labels = open(os.path.join(raw_data_dir, FileName), encoding='utf-8').readlines()

        vowels,ph_list = get_all_vowels()
        self.ph_list = ph_list

        all_temp_dict = {}
        for utterance_label in utterance_labels:
            song_info = utterance_label.split('|')
            item_name = song_info[0] #wav文件名
            temp_dict = {}

            temp_dict['wav_fn'] = f'{raw_data_dir}/wavs/{item_name}.wav'  #wav文件名
            temp_dict['txt'] = song_info[1] #歌词文本
            temp_dict['ph'] = song_info[2] #音素
            #边界识别
            temp_dict['word_boundary'] = np.array(
                [1 if x in vowels + ['AP', 'SP'] else 0 for x in song_info[2].split()])
            #音素时长
            temp_dict['ph_durs'] = [float(x) for x in song_info[5].split(" ")]
            #音高
            temp_dict['pitch_midi'] = np.array([note_to_midi(x.split("/")[0]) if x != 'rest' else 0
                                                for x in song_info[3].split(" ")])
            #音高时长
            temp_dict['midi_dur'] = np.array([float(x) for x in song_info[4].split(" ")])
            #是否连音
            temp_dict['is_slur'] = np.array([int(x) for x in song_info[6].split(" ")])
            #歌手id
            temp_dict['spk_id'] = 'staryao'
            assert temp_dict['pitch_midi'].shape == temp_dict['midi_dur'].shape == temp_dict['is_slur'].shape, \
                (temp_dict['pitch_midi'].shape, temp_dict['midi_dur'].shape, temp_dict['is_slur'].shape)

            all_temp_dict[item_name] = temp_dict

        return all_temp_dict

    # 生成训练数据
    def process_data(self):
        os.makedirs(hparams['binary_data_dir'], exist_ok=True)
        #生成歌手数据
        spk_map = set()
        for item_name in self.item_names:
            spk_name = self.items[item_name]['spk_id']
            spk_map.add(spk_name)
        spk_map = {x:i for i,x in enumerate(sorted(list(spk_map)))}
        self.spk_map = spk_map
        print("| spk_map: ", self.spk_map)
        spk_map_fn = f"{hparams['binary_data_dir']}/spk_map.json"
        json.dump(self.spk_map, open(spk_map_fn, 'w', encoding='utf-8'))

        #检查音素字典是否匹配
        ph_data = []
        for item in self.items.values():
            ph_data += item['ph'].split(' ')
        actual_phone_set = set(ph_data)
        if actual_phone_set != self.ph_list:
            print('transcriptions and dictionary mismatch.\n')

        ph_set = sorted(set(ph_data))
        # 统计训练数据音素字典覆盖情况
        self.generate_ph_summary(ph_set)

        self.phone_encoder = TokenTextEncoder(vocab_list=ph_set, replace_oov=',')

        self.process_data_split('test')
        self.process_data_split('train')
        self.process_data_split('valid')

    def process_data_split(self, prefix, multiprocess=False):
        data_dir = hparams['binary_data_dir']
        args = []
        file_path = f'{data_dir}/{prefix}'
        out_file = open(f"{file_path}.data", 'wb')
        file_byte_offsets = [0]

        lengths = []
        f0s = []
        total_sec = 0
        if self.binarization_args['with_spk_embed']:
            voice_encoder = VoiceEncoder().cuda()

        if prefix == 'valid':
            item_names = self.valid_item_names
        elif prefix == 'test':
            item_names = self.test_item_names
        else:
            item_names = self.train_item_names

        for item_name in item_names:
            meta_data = self.items[item_name]
            args.append([item_name, meta_data, self.binarization_args])

        if multiprocess:
            # code for parallel processing
            num_workers = int(os.getenv('N_PROC', os.cpu_count() // 3))
            for f_id, (_, item) in enumerate(
                    zip(tqdm(meta_data), chunked_multiprocess_run(self.process_one_item_file, args, num_workers=num_workers))):
                if item is None:
                    continue
                item['spk_embed'] = voice_encoder.embed_utterance(item['wav']) \
                    if self.binarization_args['with_spk_embed'] else None
                if not self.binarization_args['with_wav'] and 'wav' in item:
                    del item['wav']
                builder.add_item(item)
                lengths.append(item['len'])
                total_sec += item['sec']
                if item.get('f0') is not None:
                    f0s.append(item['f0'])
        else:
            # code for single cpu processing
            for i in tqdm(reversed(range(len(args))), total=len(args)):
                a = args[i]
                item = self.process_one_item_file(*a)
                if item is None:
                    continue
                item['spk_embed'] = voice_encoder.embed_utterance(item['wav']) \
                    if self.binarization_args['with_spk_embed'] else None
                if not self.binarization_args['with_wav'] and 'wav' in item:
                    del item['wav']

                #向out_file输出一条数据记录（一个wav文件）
                s = pickle.dumps(item)
                bytes = out_file.write(s)
                file_byte_offsets.append(file_byte_offsets[-1]+bytes)

                lengths.append(item['len'])
                total_sec += item['sec']
                if item.get('f0') is not None:
                    f0s.append(item['f0'])

        out_file.close()
        np.save(open(f"{file_path}.idx", 'wb'), {'offsets': file_byte_offsets})
        np.save(f'{data_dir}/{prefix}_lengths.npy', lengths)

        if len(f0s) > 0:
            f0s = np.concatenate(f0s, 0)
            f0s = f0s[f0s != 0]
            np.save(f'{data_dir}/{prefix}_f0s_mean_std.npy', [np.mean(f0s).item(), np.std(f0s).item()])
        print(f"| {prefix} total duration: {total_sec:.3f}s")

    #处理一个wav文件
    def process_one_item_file(self, item_name, meta_data, binarization_args):
        mel_path = f"{meta_data['wav_fn'][:-4]}_mel.npy"
        if os.path.exists(mel_path):
            wav = None
            mel = np.load(mel_path)
            print("load mel from npy")
        else:
            #读入WAV文件，生成mel谱特征
            global vocoder
            wav, mel = vocoder.wav2spec(meta_data['wav_fn'])

        processed_input = {
            'item_name': item_name, 'mel': mel, 'wav': wav,
            'sec': len(mel) * hparams["hop_size"] / hparams["audio_sample_rate"], 'len': mel.shape[0]
        }
        processed_input = {**meta_data, **processed_input}  # merge two dicts
        try:
            if binarization_args['with_f0']:
                f0_path = f"{meta_data['wav_fn'][:-4]}_f0.npy"
                if os.path.exists(f0_path):
                    from DiffLib.pitch_utils import f0_to_coarse
                    processed_input['f0'] = np.load(f0_path)
                    processed_input['pitch'] = f0_to_coarse(np.load(f0_path))
                else:
                    #生成基频和音高
                    gt_f0, gt_pitch_coarse = get_pitch_parselmouth(wav, mel, hparams)
                    if sum(gt_f0) == 0:
                        print("Empty **gt** f0")
                    processed_input['f0'] = gt_f0
                    processed_input['pitch'] = gt_pitch_coarse

            if binarization_args['with_txt']:
                try:
                    #根据音素字典对音素进行编码
                    phone_encoded = processed_input['phone'] = self.phone_encoder.encode(meta_data['ph'])
                except:
                    print(f"Empty phoneme")
                if binarization_args['with_align']:
                    mel2ph = np.zeros([mel.shape[0]], int)
                    startTime = 0
                    ph_durs = meta_data['ph_durs']
                    #音素和音高对齐
                    for i_ph in range(len(ph_durs)):
                        start_frame = int(startTime * hparams['audio_sample_rate'] / hparams['hop_size'] + 0.5)
                        end_frame = int((startTime + ph_durs[i_ph]) * hparams['audio_sample_rate'] / hparams['hop_size'] + 0.5)
                        mel2ph[start_frame:end_frame] = i_ph + 1
                        startTime = startTime + ph_durs[i_ph]

                    processed_input['mel2ph'] = mel2ph

        except :
            print(f"| Skip item (error:). item_name: {item_name}, wav_fn: {meta_data['wav_fn']}")
            return None
        return processed_input

    #统计训练音素数据覆盖情况
    def generate_ph_summary(self, phone_set: set):
        # Group by phonemes.
        phoneme_map = {}
        for ph in sorted(phone_set):
            phoneme_map[ph] = 0
        if hparams['use_midi']:
            for item in self.items.values():
                for ph, slur in zip(item['ph'].split(), item['is_slur']):
                    if ph not in phone_set or slur == 1:
                        continue
                    phoneme_map[ph] += 1
        else:
            for item in self.items.values():
                for ph in item['ph'].split():
                    if ph not in phone_set:
                        continue
                    phoneme_map[ph] += 1

        print('===== Phoneme Distribution Summary =====')
        for i, key in enumerate(sorted(phoneme_map.keys())):
            if i == len(phone_set) - 1:
                end = '\n'
            elif i % 10 == 9:
                end = ',\n'
            else:
                end = ', '
            print(f'\'{key}\': {phoneme_map[key]}', end=end)

        # Draw graph.
        plt.figure(figsize=(int(len(phone_set) * 0.8), 10))
        x = list(phoneme_map.keys())
        values = list(phoneme_map.values())
        plt.bar(x=x, height=values)
        plt.tick_params(labelsize=15)
        plt.xlim(-1, len(phone_set))
        for a, b in zip(x, values):
            plt.text(a, b, b, ha='center', va='bottom', fontsize=15)
        plt.grid()
        plt.title('Phoneme Distribution Summary', fontsize=30)
        plt.xlabel('Phoneme', fontsize=20)
        plt.ylabel('Number of occurrences', fontsize=20)
        filename = os.path.join(hparams['binary_data_dir'], 'phoneme_distribution.jpg')
        plt.savefig(fname=filename,
                    bbox_inches='tight',
                    pad_inches=0.25)
        print(f'| save summary to \'{filename}\'')
