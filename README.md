# Xsens + TSP balance performance and variability estimation
This repository contains the implementation of a Master's thesis: "The good imbalance, developing smart equipment with concepts from abundance theory for handstand training" [[1]](#1)

## Description
A set of scripts implementing the balance estimation model proposed in [[2]](#2), saving its data and plotting it for analysis. This branch includes a custom calibration step to allows absolute localization between devices.

## Requirements
### Dependencies
With Python 3.12.10

Run `pip install -r requirements.txt`

### Software 
- Xsens MVN Analyze
- SteamVR (null driver)

### Hardware
- Xsens MVN Link suit
- 2 Tundra trackers + 3 Base stations
- 2 Touch Sensitive Patches

#### Setup
![Setup image](images\tracking_prototype_table2_cropped.jpeg)


## Usage
Run `determineBalance.py`. 
Depending on hardware configuration some settings may have to be adjusted.

## Data
By default all data from a capture session is saved to a recordings folder with a unique identifier based on time of capture. The saved data includes timestamped pressure sensing frames and bio model metrics.
The data consists of a timeseries min distance comparison of the Center of Pressure and Extrapolated Center of Mass[[2]](#2) and a variability plot showing area within the Base of Support travelled during performance.

![data_example1](images\dist+path_custom2_more_space.png)
![data_example1](images\dist+path_custom5.png)

## References
<a id="1">[1]</a> 
link to thesis

<a id="2">[2]</a> 
Hof AL, Gazendam MG, Sinke WE. The condition for dynamic stability. J Biomech. 2005;38(1):1-8. doi:10.1016/j.jbiomech.2004.03.025
