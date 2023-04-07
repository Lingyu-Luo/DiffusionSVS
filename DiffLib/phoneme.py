
from config import hparams

PAD = "<pad>"
EOS = "<EOS>"
UNK = "<UNK>"
SEG = "|"
RESERVED_TOKENS = [PAD, EOS, UNK]
NUM_RESERVED_TOKENS = len(RESERVED_TOKENS)
PAD_ID = RESERVED_TOKENS.index(PAD)  # Normally 0
EOS_ID = RESERVED_TOKENS.index(EOS)  # Normally 1
UNK_ID = RESERVED_TOKENS.index(UNK)  # Normally 2

_initialized = False
_ALL_CONSONANTS_SET = set()
_ALL_VOWELS_SET = set()

#读入音素表
def load_phoneme_list(phoneme_path):
    _g2p_dictionary = {
        'AP': ['AP'],
        'SP': ['SP']
    }
    _set = set()
    #tmp_file_path = os.getcwd()
    #tmp_file_path = tmp_file_path + '/' + phoneme_path
    with open(phoneme_path, 'r', encoding='utf8') as _df:
        _lines = _df.readlines()
    for _line in _lines:
        _pinyin, _ph_str = _line.strip().split('\t')
        _g2p_dictionary[_pinyin] = _ph_str.split()
    for _list in _g2p_dictionary.values():
        [_set.add(ph) for ph in _list]
    _phoneme_list = sorted(list(_set))
    print('| load phone set:', _phoneme_list)

    phone_id = dict()
    id_phoneme = dict()
    for res_token in RESERVED_TOKENS:
        phone_id [ res_token ] = RESERVED_TOKENS.index(res_token)
        id_phoneme [RESERVED_TOKENS.index(res_token)] = res_token

    index_begin = len(RESERVED_TOKENS)
    for phoneme in _phoneme_list:
        phone_id[phoneme] = index_begin
        id_phoneme[index_begin] = phoneme
        index_begin = index_begin + 1

    return _phoneme_list,_g2p_dictionary,phone_id,id_phoneme,


def get_all_vowels():
    global _initialized
    if not _initialized:
        # 读入音素字典
        phone_list, pinyin2phs, phone_id, id_phoneme = load_phoneme_list(hparams['g2p_dictionary'])
        for _ph_list in pinyin2phs.values():
            _ph_count = len(_ph_list)
            if _ph_count == 0 or _ph_list[0] in ['AP', 'SP']:
                continue
            elif len(_ph_list) == 1:
                _ALL_VOWELS_SET.add(_ph_list[0])
            else:
                _ALL_CONSONANTS_SET.add(_ph_list[0])
                _ALL_VOWELS_SET.add(_ph_list[1])
        _initialized = True

    return sorted(_ALL_VOWELS_SET),phone_list