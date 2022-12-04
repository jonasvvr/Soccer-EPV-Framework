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

# pass_data = dh.read_event_tracking_data(DATA_DIR, FIELD_DIMEN, SAVE_DIR, num_files=1, max_len_data=45000)
# pass_data.to_pickle(SAVE_DIR, compression='gzip')

pass_data = pd.read_pickle(SAVE_DIR, compression='gzip')

start_passes = pass_data.loc[:, 'Event start']
end_passes = pass_data.loc[:, 'Event end']

pass_data_ = pass_data.drop(['Event start', 'Event end'], axis=1)
Xp = np.asarray(pass_data_.iloc[:, :-1])
Xd = np.asarray(end_passes)
y = pass_data_.iloc[:, -1]
Xp_train, Xp_test, Xd_train, Xd_test, y_train, y_test = train_test_split(Xp, Xd, y, test_size=0.25, stratify=y)

loss = 'binary_crossentropy'
optimizer = 'adam'
epochs = 30

soccermap = soccermap_model.SoccerMap(FIELD_DIMEN)
soccermap.compile(loss, optimizer)
print(soccermap.full.summary())





Xp_train = dh.data_to_tensor(Xp_train)
Xp_test = dh.data_to_tensor(Xp_test)
Xd_train = dh.data_to_tensor2(Xd_train)
Xd_test = dh.data_to_tensor2(Xd_test)
y_train = np.asarray(y_train).astype('float32').reshape(-1, 1)
y_test = np.asarray(y_test).astype('float32').reshape(-1, 1)

soccermap.full.fit([Xp_train, Xd_train], y_train,
                   epochs=epochs,
                   validation_data=([Xp_test, Xd_test], y_test))
