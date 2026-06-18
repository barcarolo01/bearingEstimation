import datetime as DT
import numpy as np
import csv
from build_map import *

if __name__ == "__main__":
    # Load AIS data from CSV, filter by MMSI (i.e. Vessel number), and save to output.csv
    with open('AIS_DATA/ais-2025-01-01.csv', newline='', encoding='utf-8') as f_in, \
         open('AIS_DATA/output.csv', 'w', newline='', encoding='utf-8') as f_out:
        
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if row['mmsi'] == '636018800':
                writer.writerow(row)
    
    # Read the filtered AIS data from output.csv, sort by base_date_time, and extract lat/lon coordinates
    with open('AIS_DATA/output.csv', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = sorted(reader, key=lambda r: DT.datetime.strptime(r['base_date_time'], '%Y-%m-%d %H:%M:%S'))
    lat = [row['latitude'] for row in rows]
    lon = [row['longitude'] for row in rows]


    j = np.asarray([lat,lon])
    TX_positions_coordinates = np.transpose(j)
    print(TX_positions_coordinates.shape)

    np.save("Synth/TX_Coordinates.npy",TX_positions_coordinates)
    print(f"Fetched {len(TX_positions_coordinates)} TX positions")

    fake_floater_lat = float(lat[0])+0.001
    fake_floater_lon = float(lon[0])+0.001
    fake_estimated_lat = float(lat[0])-0.001
    fake_estimated_lon = float(lon[0])-0.001

    RX_Coordinates = np.load("Synth/RX_Coordinates.npy")
    build_map(
        floaters_coordinates = RX_Coordinates,
        TX_positions_coordinates = TX_positions_coordinates,
        estimated_vessel_coordinates = [(fake_estimated_lat,fake_estimated_lon)],
        output_file="map_AIS.html"
    )

    # Tried to use the following MMSI numbers: 
    #636017837
    #316013215 #LAKE USA
    #367324580 #Back and forth
    #319469000
    #636018800 #Batimora see