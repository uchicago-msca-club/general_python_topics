"""
@author: Srihari
@date: 12/10/2018
@desc: Contains utility functions for visualisation in geospatial context
"""

import numpy as np
import folium
from folium.plugins import MarkerCluster
import webbrowser


def draw_geoheatmap(start_coord, lats, lons, mag, map_tiles=None, zoom=4, radius=15, name="Heat Map"):
    m = folium.Map(location=start_coord, tiles=map_tiles, zoom_start=zoom)
    # I am using the magnitude as the weight for the heatmap
    hm = folium.plugins.HeatMap(zip(lats, lons, mag), radius=radius, control=True)
    hm.layer_name = name
    m.add_child(hm)

    return m


def add_heatmap(m1, lats, lons, mag, radius=15, name="Heat Map"):
    # I am using the magnitude as the weight for the heatmap
    hm = folium.plugins.HeatMap(zip(lats, lons, mag), radius=radius, control=True)
    hm.layer_name = name
    m1.add_child(hm)
    return m1


def addMarkerClusters(m1, lats, lons, name="markers"):
    locations = list(zip(lats, lons))
    markers = MarkerCluster(locations=locations,
                            overlay=True,
                            control=True,
                            name=name)
    m1.add_child(markers)
    return m1


def add_choroplethmap(m1, data,
                      json_path, json_key,
                      threshold_scale,
                      data_cols=list(),
                      color="Blues", legend_name="", name="" ):
    if len(data_cols) == 0:
        data_cols = data.columns
    folium.Choropleth(geo_data=json_path,
                      data=data,
                      columns=data_cols,
                      key_on=json_key,
                      fill_color=color,
                      fill_opacity=0.7,
                      line_opacity=0.2,
                      threshold_scale=threshold_scale,
                      legend_name=legend_name,
                      name=name,
                      control=True).add_to(m1)
    return m1


def draw_choropleth_map(json_path, json_key, data, data_cols, threshold_scale,
                        zoom=4, start_coord=None, color="Blues",
                        legend_name="", name=""):
    m = folium.Map(location=start_coord, tiles="Mapbox Bright", zoom_start=zoom)
    folium.Choropleth(geo_data=json_path,
                      data=data,
                      columns=data_cols,
                      key_on=json_key,
                      fill_color=color,
                      fill_opacity=0.7,
                      line_opacity=0.2,
                      threshold_scale=threshold_scale,
                      legend_name=legend_name,
                      name=name,
                      control=True).add_to(m)

    return m


def get_th_scale(data, col, n_steps=4, transform_func=None, inv_transform_func=None):
    if transform_func is None:
        max_val = max(data[col])
        step = int(max_val / n_steps)
        return list(range(0, int(max_val + step), step))
    else:
        max_val = max(data[col].apply(transform_func))
        k = list(np.linspace(0, max_val, n_steps))
        tmp = []
        for i in range(len(k)) :
            if i == 0:
                tmp.append(0)
            else:
                tmp.append(int(np.ceil(inv_transform_func(k[i])))+1)
        return tmp


def open_map_in_browser(geomap, path):
    geomap.save(path)
    opened = webbrowser.open('file://' + path)

