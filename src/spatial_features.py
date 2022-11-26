

import numpy as np

"""
TODO angle ball  
"""


def calc_spatial_features(tracking_data, maxspeed=12, frame_diff = 0.040, field_dimen = (106.0,68.0), calc_angle_goal=False, calc_distances=False):
    
    g = np.array([field_dimen[0], field_dimen[1]/2])

    prev_xy = {}
    for i,row in tracking_data.iterrows():
        ball_xyz = row['Ball xyz']
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

            # player velocities
            vx,vy,v = calc_player_velocity(player_x, player_y, prev_xy, player_id, frame_diff)
            p_data['vx'] = vx
            p_data['vy'] = vy
            p_data['v'] = v
            # player angle to opponent's goal
            if calc_angle_goal:
                p = np.array([player_x, player_y])
                angle_goal = calc_angle_goal(p,g)
                p_data['Angle goal'] = angle_goal
            # player angle to ball
            if calc_distances:
                p = np.array([player_x, player_y])
                ball_xyz2 = ball_xyz[:-1]
                dist_ball = np.linalg.norm(p-ball_xyz2)
                p_data['Dist ball'] = dist_ball

                dist_goal = np.linalg.norm(p-g)
                p_data['Dist goal'] = dist_goal

            if (maxspeed > 0) & (v > maxspeed): 
                p_data['vx'] = np.nan
                p_data['vy'] = np.nan
                p_data['v'] = np.nan

            prev_xy[player_id] = (player_x, player_y)

    return tracking_data

def calc_player_velocity(player_x, player_y, prev_xy, player_id, frame_diff):
    vx = (player_x - prev_xy[player_id][0]) / frame_diff
    vy = (player_y - prev_xy[player_id][1]) / frame_diff
    v = np.sqrt(vx*vx + vy*vy)
    return vx,vy,v

def calc_angle_goal(p,g):
    gp = g - p
    w = gp / np.linalg.norm(gp)
    z = np.array([1,0])
    wz_norm = np.linalg.norm(np.cross(w,z))
    wz = np.dot(w,z)
    angle_goal = np.arctan2(wz_norm, wz)
    return angle_goal
