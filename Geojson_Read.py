import geojson
from descartes import PolygonPatch
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np

with open('stationbasins.geojson') as f: #Load the Geojson format file
    gj = geojson.load(f)

#Use the first set of features since the file has for every hydrological station a different area
features = gj['features'][0]

#Initiate the size of the plot
plt.clf()
ax = plt.figure(figsize=(10,10)).add_subplot(111)

#Initiate the map, "zoom" to the area specified
m = Basemap(llcrnrlon=4,llcrnrlat=43,urcrnrlon=15.,urcrnrlat=47.,resolution='i', projection='cass', lat_0 = 40, lon_0 = 0.) 

#Use the parameters for the map plot as specified below
m.drawmapboundary(fill_color='white', zorder=-1)
m.drawparallels(np.arange(44., 49., 2.), labels=[1,0,0,1], dashes=[1,1], linewidth=0.25, color='0.6',fontsize=10)
m.drawmeridians(np.arange(5.,15.,2.), labels=[1,0,0,1], dashes=[1,1], linewidth=0.25, color='0.6',fontsize=10)
m.drawcoastlines(color='0.6', linewidth=0.1)
m.drawmapboundary(fill_color='aqua')
m.fillcontinents(color='#B8860B', lake_color='aqua')
m.drawcoastlines(color ='cyan' )
m.drawcountries()


coordlist = gj['features'][2]['geometry']['rings']

#Loop to create the polygon on the map according to the coordinates  of the geojson file
for j in range(len(coordlist)):
    for k in range(len(coordlist[j])):
        coordlist[j][k][0],coordlist[j][k][1]=m(coordlist[j][k][0],coordlist[j][k][1])
    poly = {"type":"Polygon","coordinates":coordlist}#coordlist
    ax.add_patch(PolygonPatch(poly, fc=[0.2,0.3,0.8], ec=[0,0.3,0], zorder=1 ))
 
ax.axis('scaled')

#Each set of (Xi,Yi) represent the point of the stations on the map
x, y = m(11.6, 44.883335)
plt.plot(x, y, 'or', markersize=4)
plt.text(x, y, 'Pontelagoscuro', fontsize=8,weight='bold');

x1, y1 = m(10.55, 44.900002)
plt.plot(x1, y1, 'or', markersize=4)
plt.text(x1, y1, 'Boretto', fontsize=8,weight='bold');
      
x2, y2 = m(9.666667, 45.016666)
plt.plot(x2, y2, 'or', markersize=4)
plt.text(x2, y2, 'Piacenza', fontsize=8,weight='bold');

x3, y3 = m(9.1121, 45.2818)
plt.plot(x3, y3, '<y', markersize=4)
plt.text(x3, y3, 'Milan', fontsize=8,weight='bold');

x4, y4 = m(10.79, 45.16)
plt.plot(x4, y4, '<y', markersize=4)
plt.text(x4, y4, 'Mantua', fontsize=8,weight='bold');
      
x5, y5 = m(8.95664, 46)
plt.plot(x5, y5, '<y', markersize=4)
plt.text(x5, y5, 'Lugano', fontsize=8,weight='bold');

x6, y6 = m(11.50, 44.80)
plt.plot(x6, y6, '<y', markersize=4)
plt.text(x6, y6, 'Ferrara', fontsize=8,weight='bold');

x7, y7 = m(7.73, 45.03)
plt.plot(x7, y7, '<y', markersize=4)
plt.text(x7, y7, 'Turin', fontsize=8,weight='bold');

x8, y8 = m(10.42, 44.12)
plt.plot(x8, y8, '<y', markersize=4)
plt.text(x8, y8, 'Monte Cimone', fontsize=8,weight='bold');

x9, y9 = m(10.82, 45.34)
plt.plot(x9, y9, '<y', markersize=4)
plt.text(x9, y9, 'Verona', fontsize=8,weight='bold');

x10, y10 = m(11.20, 44.30)
plt.plot(x10, y10, '<y', markersize=4)
plt.text(x10, y10, 'Bologna', fontsize=8,weight='bold');

plt.show()
plt.draw()
f.close()