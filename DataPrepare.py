import os
from DiffLib.SongDesFile import SongDesFile

from config import hparams

os.makedirs(hparams['binary_data_dir'], exist_ok=True)

SongFile = SongDesFile()
SongFile.process_data()
