import pandas as pd
import numpy as np
import glob
from typing import Tuple
import gzip


"""
TODO normalize data p78 fernandez
"""

def read_event_data(DATA_DIR):

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

def read_tracking_data_single(DATA_DIR, filename=''): 
    if filename == '':
        filename = f'{DATA_DIR}/opt-tracking-25fps.txt.gz'

    columns = ['Framecount', 'Match period', 'Match status', 'Column 5', 'Ball xyz']
    column5_names = ['Object type', 'Player id', 'Shirt number', 'x', 'y']
    data = [] 
    # data.append(columns)

    with gzip.open(filename,'rt') as f: 
        for line in f: 
            line = line.split(':')
            if len(line) != 3: continue

            part1 = line[0]
            part1 = part1.split(';')
            assert len(part1) == 2, 'part1 must be len = 2'
            part1.pop(0) # remove timestamp from txt file
            part1 = part1[0].split(',')
            assert len(part1) == 3, 'part1 must be len = 3'

            part2 = line[1]
            part2 = part2.split(';')
            part2_new = []
            for p_data in part2:
                p_data = p_data.split(',')
                if len(p_data) != 5: continue
                p_data[3] = float(p_data[3])
                p_data[4] = float(p_data[4])
                p_data = dict(zip(column5_names,p_data))
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

def read_tracking_data(DATA_DIR):
    all_files = glob.glob(f'{DATA_DIR}/**/opt-tracking-25fps.txt.gz', recursive=True)

    li = []
    for filename in all_files:
        df = read_tracking_data_single(DATA_DIR, filename=filename)
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