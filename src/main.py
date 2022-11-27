import data_handler as dh
import visualization as viz
import spatial_features as spf
import game_state_representation as gsr
import pandas as pd

DATA_DIR_single = '../bru_data/data/comp-4zwgbb66rif2spcoeeol2motx/tmcl-1qtpbdbeudho5i7fu5z2lp2j8/fx-4sekj9hgwxzq3y4ih9415239w'
DATA_DIR = '../bru_data/data/comp-4zwgbb66rif2spcoeeol2motx/tmcl-1qtpbdbeudho5i7fu5z2lp2j8'

event_data_single = dh.read_event_data(DATA_DIR_single)
# event_data = dh.read_dir_event_data(DATA_DIR)

tracking_single = dh.read_tracking_data_single(DATA_DIR_single)
tracking_single = spf.calc_spatial_features(tracking_single, calc_angle_goal=False, calc_distances=False)

frame = '1000'
match_period = '1'
attacking_team = '0'
snapshot = gsr.get_tracking_data_snapshot(tracking_single, frame, attacking_team, match_period)
loc_att, loc_def, vx_att, vx_def, vy_att, vy_def = gsr.get_loc_vel_matrices(tracking_single, frame, attacking_team, match_period)
dist_b, dist_g = gsr.get_distances_matrices(tracking_single, frame, match_period)