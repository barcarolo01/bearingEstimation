
# This README is under construction
Create a `.env` file with the following environment variables

```env
SAMPLING_FREQUENCY = "96000"
NUMBER_OF_HYDROPHONES = "5"
HYDROMATE_PY_PATH = "Path/to/your/hydromate/python/hm_code/folder"
```

To run a complete simulation, the installation of Hydromate (Python version) is required. Please refers to the official page of the project for more details.


```runner.py``` 

## Inputs 
The following structures are needed to run a simulation:

* ```TX_Coordiantes```: numpy array of size (N,3)
* ```RX_Coordiantes```: numpy array of size (N,M,3)

Where N is the numper of simulation steps and M is the number of floaters (each equipped with the number of hydrohpones specified in the ```.env``` file). The last axis of both arrays should contain the triple latitude-longitude-depth.

When hydromate simulations end, two arrays will be returned, each of size (N,M): ```bearing_array``` and ```elevation_array```, which containt, respectively, the horizontal and the vertical estimated angles of arrival on n-th simulation step for the m-th floater.

Methods in ```find_points.py``` will use those information together with RX_Coordiantes to estimate the target position in every simulation step.

## Plotting
* ```build_folium_map.py``` generates an interactive HTML map using Folium library in which TX positions, RX positions and estimated positions are plotted.
* ```build_local_map.py``` generates an image of a local reference plane (x and y directions are aligned to East and North directions) in which TX positions, RX positions and estimated positions are plotted.
* ```build_local_3D.py``` generates a 3D representation of the same local reference (x and y directions are aligned to East and North directions) + the depth direction, in which TX positions, RX positions and estimated positions are plotted.