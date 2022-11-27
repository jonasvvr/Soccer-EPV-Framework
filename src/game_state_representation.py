

import numpy as np


def get_tracking_data_snapshot(tracking_data, timestamp, attacking_team, match_period, field_dimen = (106.0,68)):

    snapshot = []
    attacking_team_gk = int(attacking_team) + 2

    is_attacking = [attacking_team, attacking_team_gk]
    referee = '2'

    # Get row with timestamp
    row = tracking_data[(tracking_data['Framecount'] == timestamp) & (tracking_data['Match period'] == match_period)]
    all_player_data = row['Column 5'].iloc[0]

    # Get locations of attacking team and defending team
    attacking_team_locs = []
    defending_team_locs = []
    for player_data in all_player_data: 

        if player_data['Object type'] in is_attacking:
            attacking_team_locs.append([player_data['x'],player_data['y']])

        if (player_data['Object type'] not in is_attacking) & (player_data['Object type'] != referee):
            defending_team_locs.append([player_data['x'],player_data['y']])

    # Get goal location
    goal_loc = [field_dimen[0], field_dimen[1] / 2]
    snapshot.append(np.array(attacking_team_locs))
    snapshot.append(np.array(defending_team_locs))
    snapshot.append(np.array(row['Ball xyz'].iloc[0][:-1])) 
    snapshot.append(np.array(goal_loc))

    return snapshot

def get_loc_vel_matrices(tracking_data, timestamp, attacking_team, match_period):
    loc_att = np.empty([106,68], dtype=list)
    loc_def = np.empty([106,68], dtype=list)
    vx_att = np.empty([106,68])
    vx_def = np.empty([106,68])
    vy_att = np.empty([106,68])
    vy_def = np.empty([106,68])

    referee = '2'
    attacking_team_gk = int(attacking_team) + 2
    is_attacking = [attacking_team, attacking_team_gk]

    # Get row with timestamp
    row = tracking_data[(tracking_data['Framecount'] == timestamp) & (tracking_data['Match period'] == match_period)]
    all_player_data = row['Column 5'].iloc[0]



    for player_data in all_player_data:

        if not player_data['vx']:
            raise ValueError('Calculate velocities first')

        x = int(player_data['x'])
        y = int(player_data['y'])
        vx = player_data['vx']
        vy = player_data['vy']

        if player_data['Object type'] in is_attacking:
            # Location
            loc_att[x, y] = [player_data['x'], player_data['y']] # accurate location
            
            # Velocity
            vx_att[x,y] = vx
            vy_att[x,y] = vy

        if (player_data['Object type'] not in is_attacking) & (player_data['Object type'] != referee):  
            # Location
            loc_def[x, y] = [player_data['x'], player_data['y']]

            # Velocity
            vx_def[x,y] = vx
            vy_def[x,y] = vy


    return loc_att, loc_def, vx_att, vx_def, vy_att, vy_def


def get_distances_matrices(tracking_data, timestamp, match_period): 
    dist_b = np.empty([106,68])
    dist_g = np.empty([106,68])

    row = tracking_data[(tracking_data['Framecount'] == timestamp) & (tracking_data['Match period'] == match_period)]
    ball_xy = row['Ball xyz'].iloc[0][:-1]
    goal_xy = [106, 68/2]

    for x in range(106):
        for y in range(68): 
            a = np.array([x, y])

            distance_ball = np.linalg.norm(a - ball_xy)
            dist_b[x,y] = distance_ball

            distance_goal = np.linalg.norm(a - goal_xy)
            dist_g[x,y] = distance_goal

    return dist_b, dist_g