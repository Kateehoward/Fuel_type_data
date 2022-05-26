#basic imports
import pandas as pd
import geopandas as gpd
from netCDF4 import Dataset as NetCDFFile
import xarray as xr
import numpy as np
import sys
import os
import pickle as pkl

#reading in csv file, and making a np array of relevant coloumns
df = pd.read_csv('/g/data/er8/global_challenge/fueltype/AUS_fuel_LUT.csv')
df.head()

FTno_State = np.array(df[['FTno_State',]].values).squeeze()
FBM = np.array(df[['FBM']].values).squeeze()
# we want to find only the 'unique' values of FBM
FBM_legend = list(np.unique(df[['FBM']]))
print(FBM_legend)

#reading in and making a np array from .nc file
ds = xr.open_dataset('/g/data/er8/global_challenge/fueltype/AUS_fuel_type.nc')
#print(ds)

# one thing to note here, 
# fuel_type_image is converted to integers, since the LUT info is in integers 
# and we want to avoid issues comparing floats and ints later on
# lat and lon need to stay in float as the grid size needs the decimal places
fuel_type_image = np.array(ds.fuel_type.values, dtype=int)
lat=np.array(ds.latitude.values, dtype=float)
lon=np.array(ds.longitude.values, dtype=float)

print('fuel type shape', np.shape(fuel_type_image))
print('shape lat',np.shape(lat))

#we want variables to be the same shape as the input netcdf/fuel_type_image
FBM_image = np.zeros(fuel_type_image.shape)
print('FBM image shape', np.shape(FBM_image))

# i first ran this on the first ~10 items and printed out some info... to see what it looked like
# after i got it working, i turned off the print statements (hundreds of lines of junk/output)

# enumerate is kinda handy - idx=0,1,2... while FTno_State_idx = FTno_State[0], FTno_State[1], FTno_State[2]...
# this way, we can index FHS_s etc that lines up with FTno_State_idx
#
#for idx, FTno_State_idx in enumerate(FTno_State[0:10]):
for idx, FTno_State_idx in enumerate(FTno_State):
    #print(idx, FTno_State_idx)

    # select the locations/indices of "good pixels" == "gp" that match the FTno_State_idx
    gp = (fuel_type_image == FTno_State_idx).nonzero()
    #print(gp)

    # this is a little tricky
    # instead of storing the value of FBM (text), we are going to store the corresponding index of the unique elements
    # i can explain, but it probably needs a drawing
    # say FBM = grassland; it's also the 2nd item in the list of unique FBMs, so we store a 2 (rembering it's 0 based, so... whatever)
    FBM_image[gp] = FBM_legend.index(FBM[idx])

import h5py
#w = write
hf = h5py.File('Fuel_Type.h5', 'w')
#This creates a file object, hf, which has a bunch of associated methods. 
# One is create_dataset, which does what it says on the tin. Just provide a name for the dataset, and the numpy array.

hf.create_dataset('FBM image', data=FBM_image)
hf.create_dataset('FBM legend', data=FBM_legend)
hf.create_dataset('lat', data=lat)
hf.create_dataset('lon', data=lon)

# later on
#= assignment
#==logical != not equals > <
#mask=FBM_image==somenumber
#FBM_image[mask]=34
#FBM_image[~mask]=34 # not mask

#All we need to do now is close the file, which will write all of our work to disk.
hf.close()


import matplotlib.pyplot as plt
# similar to the gabby help example, leon_landcover_map_example.py)
classes = FBM_legend
class_colors = np.array([(123,141,245), (255,235,190), (168,0,0), (136,70,65), (0,37,115), (115,67,137), (199,215,160), (220,157,1), (57,165,9)])/255.


# anyway, we can now create a listed colour map / discrete colour map easily
import matplotlib.colors as colors
cmap = colors.ListedColormap(class_colors)

#--- think about the figure size
# how big do we want the output to be? 
output_x_size = 1600
output_y_size = 900

# pyplot specifies figure size in inches (dpi = dots per inch = anything, but 72 is a magic number from print days)
dpi = 72

#---
# mapping library
# we are working with lat/lon aka PlateCarree (https://scitools.org.uk/cartopy/docs/v0.15/crs/projections.html)
# crs is going to hold the projection that we want the map to be on
import cartopy.crs as ccrs
crs = ccrs.PlateCarree()
# the extent of the image... in map coordinates
extent = [lon.min(), lon.max(), lat.min(), lat.max()]

#---
# plotting can be complicated... main points are we want a 1x1 plot window that will output something like a 1600x900 image
# here, we specify that we want the plots to know that they are 
plt.clf()
fig, ax = plt.subplots(1, 1,
            figsize=(output_x_size/dpi, output_y_size/dpi), dpi=dpi,
            subplot_kw={'projection': crs})

# tell pyplot the map extent, and that the map extent is specified in our "map" projection - use all of it (map_extent)
map_extent = extent
ax.set_extent(map_extent, crs=crs)

img = plt.imshow(FBM_image, 
    extent=extent, 
    transform=crs, 
    interpolation='nearest', 
    cmap=cmap, 
    origin='lower')

ax.coastlines(resolution='10m', color='black', linewidth=1)
ax.gridlines(draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')

# instead, a legend is much better for this type of information
# https://stackoverflow.com/questions/25482876/how-to-add-legend-to-imshow-in-matplotlib
import matplotlib.patches as mpatches
# get the colors of the values, according to the colormap used by imshow
values = np.arange(len(classes))
img_colors = [ img.cmap(img.norm(value)) for value in values ]
# create a patch (proxy artist) for every color 
patches = [ mpatches.Patch(color=img_colors[i], label="{l}".format(l=classes[i]) ) for i in range(len(classes)) ]
# put those patched as legend-handles into the legend
# loc = 1 --> upper right (https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.legend.html)
plt.legend(handles=patches, loc=1)

plt.savefig("FBM_image_v4_aus.png")
plt.close()



#---
# plotting can be complicated... main points are we want a 1x1 plot window that will output something like a 1600x900 image
# here, we specify that we want the plots to know that they are 
plt.clf()
fig, ax = plt.subplots(1, 1,
            figsize=(output_x_size/dpi, output_y_size/dpi), dpi=dpi,
            subplot_kw={'projection': crs})

# tell pyplot the map extent, and that the map extent is specified in our "map" projection - use only the bit that covers VIC
map_extent = [140.5, 150.5, -39.5, -33.5]
ax.set_extent(map_extent, crs=crs)

img = plt.imshow(FBM_image, 
    extent=extent, 
    transform=crs, 
    interpolation='nearest', 
    cmap=cmap, 
    origin='lower')

ax.coastlines(resolution='10m', color='black', linewidth=1)
ax.gridlines(draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')

# instead, a legend is much better for this type of information
# https://stackoverflow.com/questions/25482876/how-to-add-legend-to-imshow-in-matplotlib
import matplotlib.patches as mpatches
# get the colors of the values, according to the colormap used by imshow
values = np.arange(len(classes))
img_colors = [ img.cmap(img.norm(value)) for value in values ]
# create a patch (proxy artist) for every color 
patches = [ mpatches.Patch(color=img_colors[i], label="{l}".format(l=classes[i]) ) for i in range(len(classes)) ]
# put those patched as legend-handles into the legend
# loc = 1 --> upper right (https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.legend.html)
plt.legend(handles=patches, loc=1)

plt.savefig("FBM_image_v4_vic.png")
plt.close()
