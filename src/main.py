import time

import ciso8601

import matplotlib.pyplot as plt

import data_handler as dh
import visualization as viz

DATA_DIR = '../bru_data/data/comp-4zwgbb66rif2spcoeeol2motx/tmcl-1qtpbdbeudho5i7fu5z2lp2j8'
DATA_DIR_single = '../bru_data/data/comp-4zwgbb66rif2spcoeeol2motx/tmcl-1qtpbdbeudho5i7fu5z2lp2j8/fx-vjeyiqewdovlmmq205890y6s'

FIELD_DIMEN = (112, 75)

data = dh.read_event_tracking_data(DATA_DIR, FIELD_DIMEN)

