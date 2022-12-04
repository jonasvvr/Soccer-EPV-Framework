import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import soccermap_model
import data_handler as dh

DATA_DIR = '../bru_data/data/comp-4zwgbb66rif2spcoeeol2motx/tmcl-1qtpbdbeudho5i7fu5z2lp2j8'
DATA_DIR_single = '../bru_data/data/comp-4zwgbb66rif2spcoeeol2motx/tmcl-1qtpbdbeudho5i7fu5z2lp2j8/fx-vjeyiqewdovlmmq205890y6s'
FIELD_DIMEN = (108, 72)
SAVE_DIR = '../out/pass_data3.pkl'

# pass_data = dh.read_event_tracking_data(DATA_DIR, FIELD_DIMEN, SAVE_DIR, num_files=1)
# pass_data.to_pickle(SAVE_DIR, compression='gzip')

pass_data = pd.read_pickle(SAVE_DIR, compression='gzip')
loc_attack = pass_data['Loc attack'].iloc[0]
print(pass_data)