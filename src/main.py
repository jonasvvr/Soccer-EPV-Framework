import time

import ciso8601

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data_handler as dh
import visualization as viz

DATA_DIR = '../bru_data/data/comp-4zwgbb66rif2spcoeeol2motx/tmcl-1qtpbdbeudho5i7fu5z2lp2j8'
DATA_DIR_single = '../bru_data/data/comp-4zwgbb66rif2spcoeeol2motx/tmcl-1qtpbdbeudho5i7fu5z2lp2j8/fx-vjeyiqewdovlmmq205890y6s'
FIELD_DIMEN = (112, 75)

data = dh.read_event_tracking_data(DATA_DIR, FIELD_DIMEN, num_files=0, max_len_data=45000)
data.to_pickle('../out/pass_data2.pkl')

# pass_data = pd.read_csv('../out/pass_data.csv')
# pass_data = pd.read_pickle('../out/pass_data.pkl')
# print(pass_data.iloc[0]['Loc attack'].shape)


