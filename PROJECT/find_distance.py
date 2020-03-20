import pandas as pd
import googlemaps
from itertools import tee

df = pd.read_csv('trajectory.csv')

api_key = 'AIzaSyBWlicZMkspUnL7HEWdKVy81NhynSYn2qQ'
gmaps = googlemaps.Client(key=api_key)

