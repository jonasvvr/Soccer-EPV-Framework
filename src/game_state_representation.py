import numpy as np
from scipy import spatial


def get_tracking_data_snapshot(tracking_data, timestamp, attacking_team, match_period, field_dimen):
    snapshot = []
    attacking_team_gk = str(int(attacking_team) + 3)

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
            attacking_team_locs.append([player_data['x'], player_data['y']])

        if (player_data['Object type'] not in is_attacking) & (player_data['Object type'] != referee):
            defending_team_locs.append([player_data['x'], player_data['y']])

    # Get goal location
    goal_loc = [field_dimen[0], field_dimen[1] / 2]
    snapshot.append(np.array(attacking_team_locs))
    snapshot.append(np.array(defending_team_locs))
    snapshot.append(np.array(row['Ball xyz'].iloc[0][:-1]))
    snapshot.append(np.array(goal_loc))

    return snapshot


def get_sparse_matrices(row, attacking_team, field_dimen):
    loc_att_x = np.zeros(field_dimen)
    loc_att_y = np.zeros(field_dimen)
    loc_def_x = np.zeros(field_dimen)
    loc_def_y = np.zeros(field_dimen)
    vx_att = np.zeros(field_dimen)
    vx_def = np.zeros(field_dimen)
    vy_att = np.zeros(field_dimen)
    vy_def = np.zeros(field_dimen)
    cos_angle = np.zeros(field_dimen)
    sin_angle = np.zeros(field_dimen)

    referee = '2'
    attacking_team_gk = str(int(attacking_team) + 3)
    is_attacking = [attacking_team, attacking_team_gk]

    # Get row with timestamp
    all_player_data = row['Column 5'].iloc[0]
    ball_xy = np.array(row['Ball xyz'].iloc[0][:-1])

    ball_carrier = get_ball_carrier(all_player_data, ball_xy)

    if ball_carrier is None:
        raise ValueError('Ball carrier is None')

    if 'vx' not in ball_carrier.keys():
        raise ValueError('Calculate velocities first')

    velocity_vector_bc = [ball_carrier['vx'], ball_carrier['vy']]

    for player_data in all_player_data:

        x = int(player_data['x'])
        y = int(player_data['y'])

        if (x >= field_dimen[0]) | (y >= (field_dimen[1])):  # TODO remove this
            raise ValueError(f'Out of bounds: x:{x} y:{y}')

        vx = player_data['vx']
        vy = player_data['vy']

        if player_data['Object type'] in is_attacking:
            # Location
            loc_att_x[x, y] = player_data['x']  # accurate location
            loc_att_y[x, y] = player_data['y']

            # Velocity
            vx_att[x, y] = vx
            vy_att[x, y] = vy

            # Cos angle Sin angle between velocities
            vel_vec = [vx, vy]
            sin_angle[x, y] = get_sine_angle(vel_vec, velocity_vector_bc)
            cos_angle[x, y] = get_cosine_angle(vel_vec, velocity_vector_bc)

        if (player_data['Object type'] not in is_attacking) & (player_data['Object type'] != referee):
            # Location
            loc_def_x[x, y] = player_data['x']
            loc_def_y[x, y] = player_data['y']

            # Velocity
            vx_def[x, y] = vx
            vy_def[x, y] = vy

    return loc_att_x, loc_att_y, loc_def_x, loc_def_y, vx_att, vx_def, vy_att, vy_def, cos_angle, sin_angle


def get_dense_matrices(row, field_dimen):
    dist_b = np.zeros(field_dimen)
    dist_g = np.zeros(field_dimen)
    angle_ball_sin = np.zeros(field_dimen)
    angle_ball_cos = np.zeros(field_dimen)
    angle_goal_sin = np.zeros(field_dimen)
    angle_goal_cos = np.zeros(field_dimen)
    angle_goal_rad = np.zeros(field_dimen)

    ball_xy = np.array(row['Ball xyz'].iloc[0][:-1])
    goal_xy = np.array([field_dimen[0], field_dimen[1] / 2])

    for x in range(field_dimen[0]):
        for y in range(field_dimen[1]):
            a = np.array([x, y])

            distance_ball = np.linalg.norm(a - ball_xy)
            dist_b[x, y] = distance_ball

            distance_goal = np.linalg.norm(a - goal_xy)
            dist_g[x, y] = distance_goal

            cos_ball = get_cosine_angle(a, ball_xy)
            sin_ball = get_sine_angle(a, ball_xy)
            angle_ball_sin[x, y] = sin_ball
            angle_goal_cos[x, y] = cos_ball

            cos_goal = get_cosine_angle(a, goal_xy)
            sin_goal = get_sine_angle(a, goal_xy)
            angle_goal_sin[x, y] = sin_goal
            angle_goal_cos[x, y] = cos_goal

            angle_goal_rad[x, y] = get_angle_rad(a, goal_xy)

    return dist_b, dist_g, angle_ball_sin, angle_ball_cos, angle_goal_sin, angle_goal_cos, angle_goal_rad


def get_cosine_angle(vec1, vec2):
    dot = np.dot(vec1, vec2)
    vec1_mag = np.linalg.norm(vec1)
    vec2_mag = np.linalg.norm(vec2)
    denom = vec1_mag * vec2_mag
    if denom == 0:
        return 0
    return dot / denom


def get_sine_angle(vec1, vec2):
    cross = np.cross(vec1, vec2)
    cross_norm = np.linalg.norm(cross)
    vec1_mag = np.linalg.norm(vec1)
    vec2_mag = np.linalg.norm(vec2)
    denom = vec1_mag * vec2_mag
    if denom == 0:
        return 0
    return cross_norm / denom


def get_angle_rad(vec1, vec2):
    sine = get_sine_angle(vec1, vec2)
    return np.arcsin(sine)


def get_ball_carrier(all_player_data, ball_xy):
    coords = []
    dict_ = {}

    for p_data in all_player_data:
        coord = (p_data['x'], p_data['y'])
        coords.append(coord)
        dict_[coord] = p_data

    tree = spatial.KDTree(coords)
    (_, idx) = tree.query(ball_xy)

    try:
        closest = coords[idx]
        closest_player = dict_[closest]
    except Exception as e:
        print(e)
        return None

    return closest_player


def get_game_state_representation(row, attacking_team, field_dimen):
    loc_att_x, loc_att_y, loc_def_x, loc_def_y, vx_att, vx_def, vy_att, vy_def, cos_angle, sin_angle = get_sparse_matrices(
        row, attacking_team,
        field_dimen)
    dist_b, dist_g, angle_ball_sin, angle_ball_cos, angle_goal_sin, angle_goal_cos, angle_goal_rad = get_dense_matrices(
        row, field_dimen)

    return [loc_att_x, loc_att_y, loc_def_x, loc_def_y, vx_att, vx_def, vy_att, vy_def, dist_b, dist_g, angle_ball_sin,
            angle_ball_cos, angle_goal_sin, angle_goal_cos, angle_goal_rad,
            sin_angle, cos_angle]
