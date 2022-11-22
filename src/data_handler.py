import pandas as pd
import numpy as np
import glob
from typing import Tuple

def read_event_data(DATA_DIR):

    # all_files = glob.glob(f'{DATA_DIR}/**/*events.json.gz', recursive=True)

    filename = f'{DATA_DIR}/ma3-match-events.json.gz'

    df = pd.read_json(
        filename,
        compression='gzip'
    )

    events = pd.DataFrame(df['liveData']['event'])
    events = events.drop(['contestantId', 'timeStamp', 'lastModified', 'playerId', 'id', 'eventId'], axis=1)

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
        df = df.drop(['contestantId', 'timeStamp', 'lastModified', 'playerId', 'id', 'eventId'], axis=1)
    
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

def convert_to_middle_origin(xy: Tuple[float,float], field_dimen: Tuple[float,float] = (106.0,68)):
    return (xy[0] - (field_dimen[0]/2), xy[1] - (field_dimen[1]/2))

def scale_coord(xy: Tuple[float,float], field_dimen: Tuple[float,float] = (106.0,68)):
    return ( (xy[0]/100) * field_dimen[0] , (xy[1] / 100) * field_dimen[1] )

def conv_scale(xy: Tuple[float,float], field_dimen: Tuple[float,float] = (106.0,68)):
    return convert_to_middle_origin(scale_coord(xy,field_dimen),field_dimen)
    

