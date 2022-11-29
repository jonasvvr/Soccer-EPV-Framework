from matplotlib import pyplot as plt
import numpy as np
import data_handler as dh
from LaurieOnTracking.Metrica_Viz import plot_pitch


def plot_events(events, field_dimen, figax=None, indicators=['Marker', 'Arrow'], color='r',
                marker_style='o', alpha=0.5, annotate=False):
    if figax is None:
        fig, ax = plot_pitch(field_dimen=field_dimen)
    else:
        fig, ax = figax

    for i, row in events.iterrows():
        if row['typeId'] != 1:
            # print('Event is not a pass')
            continue
        x = row['x']
        y = row['y']
        endx = float(dh.get_qualifier_value(row['qualifier'], 140))
        endy = float(dh.get_qualifier_value(row['qualifier'], 141))

        # scale and convert to new origin 
        (x, y) = dh.conv_scale((x, y), field_dimen)
        (endx, endy) = dh.conv_scale((endx, endy), field_dimen)

        if 'Marker' in indicators:
            ax.plot(x, y, color + marker_style, alpha=alpha)
        if 'Arrow' in indicators:
            ax.annotate('', xy=(endx, endy), xytext=(x, y),
                        alpha=alpha,
                        arrowprops=dict(alpha=alpha, width=0.5, headlength=4.0, headwidth=4.0, color=color),
                        annotation_clip=False)
        if annotate:
            textstring = 'Pass' + ': ' + row['playerName']
            ax.text(x, y, textstring, fontsize=10, color=color)
    return fig, ax


def plot_frame(tracking_frame, field_dimen, figax=None, include_player_velocities=False,
               PlayerMarkerSize=10, PlayerAlpha=0.7, annotate=False):
    if figax is None:
        fig, ax = plot_pitch(field_dimen=field_dimen)
    else:
        fig, ax = figax

    data = tracking_frame['Column 5']
    ball_xyz = tracking_frame['Ball xyz']

    ball_xy = dh.convert_to_middle_origin((ball_xyz[0], ball_xyz[1]), field_dimen)

    for row in data:
        object_type = row['Object type']
        player_id = row['Player id']
        shirt_number = row['Shirt number']
        xy = dh.convert_to_middle_origin((row['x'], row['y']), field_dimen)
        color = getColor(object_type)
        ax.plot(xy[0], xy[1], color=color, marker='o', markersize=PlayerMarkerSize, alpha=PlayerAlpha)

        if include_player_velocities:
            vx = row['vx']
            vy = row['vy']
            ax.quiver(xy[0], xy[1], vx, vy, color=color, scale_units='inches', scale=10., width=0.0015, headlength=5,
                      headwidth=3, alpha=PlayerAlpha)

        if annotate:
            ax.text(xy[0] + 0.7, xy[1] + 0.7, shirt_number, fontsize=10, color=color)

    ax.plot(ball_xy[0], ball_xy[1], 'k', marker='o', markersize=6, alpha=1.0, linewidth=0)
    return fig, ax


def getColor(object_type):
    match object_type:
        case '0':
            return 'r'
        case '1':
            return 'b'
        case '2':
            return 'c'
        case '3':
            return 'r'
        case '4':
            return 'b'
        case _:
            raise ValueError(f'Object type has an invalid value: {object_type}')
