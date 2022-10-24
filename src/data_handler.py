import pandas as pd
import csv as csv

def read_event_data(dir: str, game_id: int):
    event_file = f'/sample_game_{game_id}/Sample_Game_{game_id}_RawEventsData.csv'
    events = pd.read_csv(f'{dir}/{event_file}')
    return events

def read_tracking_data(dir: str, game_id:int, team: str): 
    tracking_data_file = f'/sample_game_{game_id}/Sample_Game_{game_id}_RawTrackingData_{team}_Team.csv'
    csvfile =  open(f'{dir}/{tracking_data_file}', 'r') 
    reader = csv.reader(csvfile)

    teamnamefull = next(reader)[3].lower()
    print("Reading team: %s" % teamnamefull)

    
    jerseys = [x for x in next(reader) if x != ''] 
    columns = next(reader)
    for i, j in enumerate(jerseys): 
        columns[i*2+3] = f'{team}_{j}_x'
        columns[i*2+4] = f'{team}_{j}_y'
    columns[-2] = "ball_x"
    columns[-1] = "ball_y"

    # Second: read in tracking data and place into pandas Dataframe
    tracking = pd.read_csv(f'{dir}/{tracking_data_file}', names=columns, index_col='Frame', skiprows=3)
    return tracking

def to_metric_coordinates(data,field_dimen=(106.,68.) ):

    x_columns = [c for c in data.columns if c[-1].lower()=='x']
    y_columns = [c for c in data.columns if c[-1].lower()=='y']
    data[x_columns] = ( data[x_columns]-0.5 ) * field_dimen[0]
    data[y_columns] = -1 * ( data[y_columns]-0.5 ) * field_dimen[1]
    
    return data