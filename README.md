This codes creates a vertical profile of wind speed by height for Jülich. It uses data of the Aeolus satellite mission and compares them with the JOYCE observation data and the ICON model data. The wind speed profiles only show the windspeed in the horizontal line of sight (HLOS) of the satellite.
# Usage
```
python3 vert_profile.py
```

# Customization
The image path, data path and used satellite orbit can be changed in the file 'vert_profile.py' in the section:
```python
################# change here! #################

orbit = '3082'
path = '/work/marcus_mueller/aeolus/'
imagePath = "/work/marcus_mueller/aeolus/3082/plots2/"

################################################
```
