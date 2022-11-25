

import numpy as np


def calc_player_velocities(tracking_data, maxspeed=12, frame_diff = 0.040):
    

    prev_xy = {}
    for i,row in tracking_data.iterrows():

        row = row['Column 5']

        for p_data in row: 
            player_id = p_data['Player id']
            player_x = p_data['x']
            player_y = p_data['y']
            
            if player_id not in prev_xy: 
                prev_xy[player_id] = (player_x, player_y)
                p_data['vx'] = 0.0
                p_data['vy'] = 0.0
                p_data['v'] = 0.0
                continue

            vx = (player_x - prev_xy[player_id][0]) / frame_diff
            vy = (player_y - prev_xy[player_id][1]) / frame_diff
            v = np.sqrt(vx*vx + vy*vy)
            p_data['vx'] = vx
            p_data['vy'] = vy
            p_data['v'] = v

            if (maxspeed > 0) & (v > maxspeed): 
                p_data['vx'] = np.nan
                p_data['vy'] = np.nan
                p_data['v'] = np.nan

            prev_xy[player_id] = (player_x, player_y)

    return tracking_data
