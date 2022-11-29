import numpy as np


def calc_spatial_features(row, index, tracking_data, maxspeed=12, frame_diff=0.100):
    if index != 0:
        row_prev = tracking_data.iloc[index - 1]
        prev_player_data = row_prev['Column 5']

    player_data = row['Column 5'].iloc[0]

    for i in range(len(player_data)):
        p_data = player_data[i]

        if 'row_prev' not in locals():
            p_data['vx'] = 0.0
            p_data['vy'] = 0.0
            p_data['v'] = 0.0
            player_data[i] = p_data
            continue

        prev_p_data = prev_player_data[i]

        player_x = p_data['x']
        player_y = p_data['y']

        prev_x = prev_p_data['x']
        prev_y = prev_p_data['y']

        # player velocities
        vx, vy, v = calc_player_velocity(player_x, player_y, prev_x, prev_y, frame_diff)
        p_data['vx'] = vx
        p_data['vy'] = vy
        p_data['v'] = v

        if (maxspeed > 0) & (v > maxspeed):
            p_data['vx'] = np.nan
            p_data['vy'] = np.nan
            p_data['v'] = np.nan

        player_data[i] = p_data

    return row


def calc_player_velocity(player_x, player_y, prev_x, prev_y, frame_diff):
    vx = (player_x - prev_x) / frame_diff
    vy = (player_y - prev_y) / frame_diff
    v = np.sqrt(vx * vx + vy * vy)
    return vx, vy, v
