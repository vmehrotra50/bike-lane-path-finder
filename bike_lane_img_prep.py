import os

# Prepare the images


i = 0
bike_lanes_dir = 'C:/Users/varun/Documents/GitHub/bike-lane-path-finder/bike_lanes'

'''
for file in os.listdir(bike_lanes_dir):
    new_name = "BIKE_LANE_" + str(i)
    path = os.path.join(bike_lanes_dir, file)
    os.rename(path, os.path.join(bike_lanes_dir, new_name))
    i += 1
    print(file)
'''

no_bike_lanes_dir = 'C:/Users/varun/Documents/GitHub/bike-lane-path-finder/no_bike_lanes'

for file in os.listdir(no_bike_lanes_dir):
    new_name = "NOT_BIKE_LANE_" + str(i)
    path = os.path.join(no_bike_lanes_dir, file)
    os.rename(path, os.path.join(no_bike_lanes_dir, new_name))
    i += 1
    print(file)