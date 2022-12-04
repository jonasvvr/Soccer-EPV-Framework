import pandas as pd
import numpy as np
import glob
from typing import Tuple
import gzip

import tensorflow as tf

import game_state_representation as gsr
import spatial_features as spf
from tqdm import tqdm


def read_event_data(DATA_DIR, filename=''):
    if filename == '':
        filename = f'{DATA_DIR}/ma3-match-events.json.gz'

    df = pd.read_json(
        filename,
        compression='gzip'
    )

    events = pd.DataFrame(df['liveData']['event'])
    return events


def read_dir_event_data(DATA_DIR):
    all_files = glob.glob(f'{DATA_DIR}/**/*events.json.gz', recursive=True)

    li = []
    for filename in all_files:
        df = pd.read_json(
            filename,
            compression='gzip'
        )

        df = pd.DataFrame(df['liveData']['event'])
        li.append(df)

    return pd.concat(li, axis=0, ignore_index=True)


def read_tracking_data(DATA_DIR, filename=''):
    if filename == '':
        filename = f'{DATA_DIR}/opt-tracking-10fps.txt.gz'

    columns = ['Timestamp', 'Framecount', 'Match period', 'Match status', 'Column 5', 'Ball xyz']
    column5_names = ['Object type', 'Player id', 'Shirt number', 'x', 'y']
    data = []
    # data.append(columns)

    with gzip.open(filename, 'rt') as f:
        for line in f:
            line = line.split(':')
            if len(line) != 3: continue

            part1 = line[0]
            part1 = part1.split(';')
            assert len(part1) == 2, 'part1 must be len = 2'
            part1.extend(part1[1].split(','))
            part1.pop(1)
            assert len(part1) == 4, 'part1 must be len = 4'

            part2 = line[1]
            part2 = part2.split(';')
            part2_new = []
            for p_data in part2:
                p_data = p_data.split(',')
                if len(p_data) != 5: continue
                p_data[3] = float(p_data[3])
                p_data[4] = float(p_data[4])
                p_data = dict(zip(column5_names, p_data))
                part2_new.append(p_data)

            part3 = line[2]
            part3 = part3.split(';')
            if len(part3) == 2:
                part3.pop(1)
            part3 = part3[0].split(',')
            assert len(part3) == 3, 'part3 must be len = 3'
            part3 = [float(i) for i in part3]

            full = part1
            full.append(part2_new)
            full.append(part3)

            data.append(full)

    data = pd.DataFrame(data, columns=columns)

    return data


def read_dir_tracking_data(DATA_DIR):
    all_files = glob.glob(f'{DATA_DIR}/**/opt-tracking-10fps.txt.gz', recursive=True)

    li = []
    for filename in all_files:
        df = read_tracking_data(DATA_DIR, filename=filename)
        li.append(df)

    return pd.concat(li, axis=0, ignore_index=True)


def find_qualifier(list: list, id: int):
    dict = next((item for item in list if item['qualifierId'] == id), None)
    if dict == None:
        raise ValueError("Qualifier not found")
    return dict


def get_qualifier_value(list: list, id: int):
    dict = find_qualifier(list, id)
    try:
        return dict['value']
    except:
        raise ValueError("Qualifier has no value")


def convert_to_middle_origin(xy: Tuple[float, float], field_dimen):
    return xy[0] - (field_dimen[0] / 2), xy[1] - (field_dimen[1] / 2)


def scale_coord(xy: Tuple[float, float], field_dimen):
    return (xy[0] / 100) * field_dimen[0], (xy[1] / 100) * field_dimen[1]


def conv_scale(xy: Tuple[float, float], field_dimen):
    return convert_to_middle_origin(scale_coord(xy, field_dimen), field_dimen)


def scale_event_coords(event, field_dimen):
    if 'endx' not in event.keys():
        raise ValueError('End coords do not exist')

    event['x'], event['y'] = scale_coord((event['x'], event['y']), field_dimen)
    event['endx'], event['endy'] = scale_coord((event['endx'], event['endy']), field_dimen)

    return event


def read_event_tracking_data(DATA_DIR, field_dimen, save_dir, fps=10, tracking_accuracy=1.3, num_files=0,
                             max_len_data=50000,
                             save_every=5):
    column_names = ['Event start', 'Event end', 'Loc attack', 'Loc defend',
                    'vx attack', 'vx defend',
                    'vy attack', 'vy defend',
                    'Distance ball', 'Distance goal', 'Angle ball sin', 'Angle ball cos', 'Angle goal sin',
                    'Angle goal cos', 'Angle goal rad',
                    'Ball carrier sine', 'Ball carrier cosine', 'Outcome']

    all_event_files = glob.glob(f'{DATA_DIR}/**/*events.json.gz', recursive=True)
    all_tracking_files = glob.glob(f'{DATA_DIR}/**/opt-tracking-{fps}fps.txt.gz', recursive=True)

    data = []

    for i in range(len(all_event_files)):
        if (i % save_every == 0) & (i != 0):
            data_df = pd.concat(data, axis=0, ignore_index=True)
            data_df.to_pickle(save_dir)

        print(f'Reading file {i + 1}...')
        event_file = all_event_files[i]
        tracking_file = all_tracking_files[i]

        tracking_data = read_tracking_data(DATA_DIR, filename=tracking_file)
        event_data = read_event_data(DATA_DIR, filename=event_file)

        if tracking_data.shape[0] == 0 | event_data.shape[0] == 0:
            continue

        passing_events = event_data[event_data['typeId'] == 1]

        k = 0
        for _, pass_event in tqdm(passing_events.iterrows()):

            match_period = str(pass_event['periodId'])
            timestamp = get_frame(pass_event['timeMin'], pass_event['timeSec'], match_period)

            # Get tracking data frame
            try:
                row = tracking_data[
                    (tracking_data['Framecount'] == timestamp) & (tracking_data['Match period'] == match_period)]
                index = tracking_data[(tracking_data['Framecount'] == timestamp) & (
                        tracking_data['Match period'] == match_period)].index.values[0]
            except Exception as e:
                print(f'Exception occurred with getting tracking data row and index: {e}')
                continue

            if distance_ball_ball_carier(row) > tracking_accuracy:
                continue

            if out_of_bounds(pass_event, field_dimen):
                # print('Out of bounds')
                continue

            attacking_team = get_attacking_team(pass_event, row)

            if attacking_team is None:
                print('attacking_team is None')
                continue

            row = spf.calc_spatial_features(row, index, tracking_data)
            try:
                game_state_rep = gsr.get_game_state_representation(row, attacking_team,
                                                                   field_dimen)  # TODO maybe after saving
            except ValueError as e:
                print(e)
                continue

            pass_event_data = handle_pass_event(pass_event, field_dimen)
            data.append(get_event_data(pass_event_data, game_state_rep, column_names))  # TODO maybe after saving

            k += 1
            if len(data) == max_len_data:
                break

        if (i == (num_files - 1)) & (num_files != 0):
            break
        if len(data) == max_len_data:
            break
    return pd.concat(data, axis=0, ignore_index=True)


def handle_pass_event(pass_event, field_dimen):
    start_pass = np.zeros((field_dimen[0], field_dimen[1]), dtype=np.int32)
    dest_pass = np.zeros((field_dimen[0], field_dimen[1]), dtype=np.int32)
    x = pass_event['x']
    y = pass_event['y']
    endx = float(get_qualifier_value(pass_event['qualifier'], 140))
    endy = float(get_qualifier_value(pass_event['qualifier'], 141))
    outcome = pass_event['outcome']
    start_pass[int(x), int(y)] = 1 if outcome == 1 else 0
    dest_pass[int(endx), int(endy)] = 1 if outcome == 1 else 0

    return [start_pass, dest_pass, outcome]


def distance_ball_ball_carier(tracking_data_row):
    ball_xy = np.array(tracking_data_row['Ball xyz'].iloc[0][:-1])
    ball_carrier = gsr.get_ball_carrier(tracking_data_row['Column 5'].iloc[0], ball_xy)
    ball_carrier_xy = np.array([ball_carrier['x'], ball_carrier['y']])
    return np.linalg.norm(ball_xy - ball_carrier_xy)


def get_event_data(pass_event_data, game_state_rep, column_names):
    event = pd.DataFrame(columns=column_names)

    for i in range(len(column_names)):
        if i <= 1:
            event.loc[0, column_names[i]] = pass_event_data[i]
            continue
        if i == (len(column_names) - 1):
            event.loc[0, column_names[i]] = pass_event_data[2]
            continue
        if i > 1:
            event.loc[0, column_names[i]] = game_state_rep[i - 2]
            continue

    return event


def get_frame(time_min, time_sec, match_period):
    if match_period == '2':
        time_2 = time_min - 45
        return str((time_2 * 60 + time_sec) * 1000)

    return str((time_min * 60 + time_sec) * 1000)


def get_attacking_team(event, row):
    player_id = event['playerId']
    list = row['Column 5'].iloc[0]

    dict = next((item for item in list if item['Player id'] == player_id), None)

    if dict is None:
        return None

    object_type = dict['Object type']

    match object_type:
        case '0':
            return '0'
        case '1':
            return '1'
        case '2':
            raise ValueError('Returned referee')
        case '3':
            return '0'
        case '4':
            return '1'
        case _:
            raise ValueError(f'Object type has an invalid value: {object_type}')


def convert_data_to_tensor(X):
    for (x, y), elem in np.ndenumerate(X):
        X[x, y] = tf.convert_to_tensor(elem)

    return X


def out_of_bounds(pass_event, field_dimen):
    endx = int(float(get_qualifier_value(pass_event['qualifier'], 140)))
    endy = int(float(get_qualifier_value(pass_event['qualifier'], 141)))
    x = int(pass_event['x'])
    y = int(pass_event['y'])

    if (endx >= field_dimen[0]) | (endy >= (field_dimen[1])):
        return True
    if (endx < 0) | (endy < 0):
        return True
    if (x >= field_dimen[0]) | (y >= (field_dimen[1])):
        return True
    if (x < 0) | (y < 0):
        return True
    return False

def data_to_tensor(data):
    for (x, y), elem in np.ndenumerate(data):
        data[x, y] = tf.convert_to_tensor(elem)

    stacked_rows = [tf.stack(row, axis=2) for row in data]
    return tf.stack(stacked_rows)


def data_to_tensor2(data):
    stacked_rows = [tf.convert_to_tensor(row) for row in data]
    return tf.stack(stacked_rows)