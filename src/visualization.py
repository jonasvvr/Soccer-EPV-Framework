import data_handler as dh

def plot_events( events, plot_pitch,figax=None, field_dimen = (106.0,68), indicators = ['Marker','Arrow'], color='r', marker_style = 'o', alpha = 0.5, annotate=False):
    
    if figax is None:
        fig,ax = plot_pitch(field_dimen=field_dimen)
    else: 
        fig,ax = figax

    for i, row in events.iterrows(): 
        if row['typeId'] != 1:
            # print('Event is not a pass')
            continue
        x = row['x']
        y = row['y']
        endx = float(dh.get_qualifier_value(row['qualifier'],140))
        endy = float(dh.get_qualifier_value(row['qualifier'],141))

        # scale and convert to new origin 
        (x,y) = dh.conv_scale((x,y), field_dimen)
        (endx,endy) = dh.conv_scale((endx,endy), field_dimen)
        
        if 'Marker' in indicators: 
            ax.plot( x, y, color+marker_style, alpha=alpha )
        if 'Arrow' in indicators: 
            ax.annotate('', xy = (endx, endy), xytext=(x, y),
                alpha=alpha, arrowprops=dict(alpha=alpha,width=0.5,headlength=4.0,headwidth=4.0,color=color),annotation_clip=False)
        if annotate:
            textstring = 'Pass' + ': ' + row['playerName']
            ax.text( x, y, textstring, fontsize=10, color=color)
    return fig,ax
