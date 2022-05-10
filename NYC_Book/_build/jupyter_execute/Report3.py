#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure

# Libaries
from scipy.stats.stats import pearsonr   
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, precision_score
from sklearn import tree
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from bokeh.plotting import figure as b_figure
from bokeh.models import ColumnDataSource
from bokeh.models import Legend, LegendItem
from bokeh.models import Range1d
from bokeh.palettes import Spectral6
from bokeh.io import output_file, show
from bokeh.io import output_notebook
from bokeh.layouts import layout
from bokeh.models.widgets import Tabs, Panel
from bokeh.io import curdoc
import matplotlib.lines as mlines
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import plotly.express as px
from urllib.request import urlopen
import json
import geopandas as gpd
from IPython.core.display import display, HTML
from plotly.offline import download_plotlyjs, init_notebook_mode, plot as px_plot
config={'showLink': False, 'displayModeBar': False}
from plotly.subplots import make_subplots

init_notebook_mode()

import math
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Read data
data = pd.read_csv("NYCgov_Poverty_Measure_Data__2015_.csv")

# Adding column for age-group
bins= np.arange(0,110,10)
labels = ['(' + str(x)+'-'+str(x+10)+']' for x in bins[:-1]]
labels[-1] = labels[-1].replace("100]", "inf)")
data['AgeGroup'] = pd.cut(data['AGEP'], bins=bins, labels=labels, right=True)

# Create dataframe with only adults (finished school)
data_adult = data[data['AGEP'] > 24]

# Mapping of different columns 
EducAttain_map = {1:'Less than High School',2:'High School Degree',3:'Some College',4:'Bachelor\'s Degree or higher'}
SEX_map = {1: 'Male', 2:'Female'}
Boro_map = {1:'Bronx', 2:'Brooklyn',3:'Manhattan',4:'Queens',5:'Staten Island'}
Off_Pov_Stat_map = {1:'In Poverty',2:'Not in Poverty'}
Ethnicity_map = {1:'Non-Hispanic White',2:'Non-Hispanic Black',3:'Non-Hispanic Asian',
    4:'Hispanic, Any Race',5:'Other Race/Ethnic Group'}

# Adding column of total income 
temp_col = (data_adult['INTP_adj'] + data_adult['SEMP_adj'] + data_adult['WAGP_adj'] +data_adult['RETP_adj'])/1000
data_adult.insert(1, "Total_income", temp_col, True)


# In[3]:


## Defining functions for finding bin size
def bins_sturges(n):
    return int(1 + np.ceil(math.log(n)))

def bins_freedamn_diaconis(data):
    q3 = np.quantile(data, 0.75)
    q1 = np.quantile(data, 0.25)
    b_w = 2*(q3-q1)/(np.cbrt(len(data)))
    if b_w == 0: 
        b_w = 1
    bins = int(np.ceil((max(data) - min(data))/b_w))
    return bins 


# In[4]:


## Heatmap for distribution of education level in different boroughs
gdf = gpd.read_file('https://raw.githubusercontent.com/dwillis/nyc-maps/master/boroughs.geojson')
gdf.to_crs(epsg=4326, inplace=True)
gdf.set_index('BoroName', inplace=True)
gdf['BoroCode'] = [5,4,2,3,1]
gdf.sort_index(inplace=True)
# Making subplots for each education level
fig = make_subplots(rows=2, cols=2,
                    subplot_titles=list(EducAttain_map.values()),
                    specs=[[{'type':"mapbox"}, {'type':"mapbox"}],[{'type':"mapbox"}, {'type':"mapbox"}]],
                    horizontal_spacing= 0.05,
                    vertical_spacing=0.08) 
list_edu = list(EducAttain_map.keys())
# Subplot of less than highschool 
i = list_edu[0]
X_grouped = round(data_adult[data_adult['EducAttain'] == i].groupby(['Boro']).count()/data_adult.groupby(['Boro']).count(),2)*100
X_grouped['BoroName'] = ['Bronx','Brooklyn','Manhattan','Queens','Staten Island']
X_grouped.set_index('BoroName',inplace=True)
att = 'Total_income'
label_text = 'Percentage of ' +  EducAttain_map[i] + ':'
fig1 = px.choropleth_mapbox(X_grouped[att].reset_index(), geojson=gdf['geometry'], locations=gdf.index, color='Total_income',
                        color_continuous_scale="Viridis",
                        range_color=(X_grouped[att].min(),X_grouped[att].max()),
                        mapbox_style="carto-positron",
                        zoom=8.5, center = {"lat": 40.730610, "lon": -73.935242},
                        opacity=0.5,
                        labels={att: label_text}
                        )
# Subplot of highschool 
i = list_edu[1]
X_grouped = round(data_adult[data_adult['EducAttain'] == i].groupby(['Boro']).count()/data_adult.groupby(['Boro']).count(),2)*100
X_grouped['BoroName'] = ['Bronx','Brooklyn','Manhattan','Queens','Staten Island']
X_grouped.set_index('BoroName',inplace=True)
att = 'Total_income'
label_text = 'Percentage of ' +  EducAttain_map[i] + ':'
fig2 = px.choropleth_mapbox(X_grouped[att].reset_index(), geojson=gdf['geometry'], locations=gdf.index, color='Total_income',
                        color_continuous_scale="Viridis",
                        range_color=(X_grouped[att].min(),X_grouped[att].max()),
                        mapbox_style="carto-positron",
                        zoom=8.5, center = {"lat": 40.730610, "lon": -73.935242},
                        opacity=0.5,
                        labels={att:label_text}
                        )
# Subplot of some college 
i = list_edu[2]
X_grouped = round(data_adult[data_adult['EducAttain'] == i].groupby(['Boro']).count()/data_adult.groupby(['Boro']).count(),2)*100
X_grouped['BoroName'] = ['Bronx','Brooklyn','Manhattan','Queens','Staten Island']
X_grouped.set_index('BoroName',inplace=True)
att = 'Total_income'
label_text = 'Percentage of ' +  EducAttain_map[i] + ':'
fig3 = px.choropleth_mapbox(X_grouped[att].reset_index(), geojson=gdf['geometry'], locations=gdf.index, color='Total_income',
                        color_continuous_scale="Viridis",
                        range_color=(X_grouped[att].min(),X_grouped[att].max()),
                        mapbox_style="carto-positron",
                        zoom=8.5, center = {"lat": 40.730610, "lon": -73.935242},
                        opacity=0.5,
                        labels={att:label_text}
                        )
# Subplot of bachelor's or higher
i = list_edu[3]
X_grouped = round(data_adult[data_adult['EducAttain'] == i].groupby(['Boro']).count()/data_adult.groupby(['Boro']).count(),2)*100
X_grouped['BoroName'] = ['Bronx','Brooklyn','Manhattan','Queens','Staten Island']
X_grouped.set_index('BoroName',inplace=True)
att = 'Total_income'
label_text = 'Percentage of ' +  EducAttain_map[i] + ':'
fig4 = px.choropleth_mapbox(X_grouped[att].reset_index(), geojson=gdf['geometry'], locations=gdf.index, color='Total_income',
                        color_continuous_scale="Viridis",
                        range_color=(X_grouped[att].min(),X_grouped[att].max()),
                        mapbox_style="carto-positron",
                        zoom=8.5, center = {"lat": 40.730610, "lon": -73.935242},
                        opacity=0.5,
                        labels={att:label_text}
                        )
# Layout and show heatmaps
fig.add_trace(fig1['data'][0], row=1, col=1)
fig.add_trace(fig2['data'][0], row=1, col=2)
fig.add_trace(fig3['data'][0], row=2, col=1)
fig.add_trace(fig4['data'][0], row=2, col=2)
fig.update_mapboxes(style='carto-positron',
                    zoom=7.5, center = {"lat": 40.730610, "lon": -73.935242})
fig.update_layout(autosize=False, width = 800, height=600, title_text='Distribution of education level for each borough', title_x=0.5)
fig.layout.coloraxis.colorbar.title = '%'
px_plot(fig, filename = 'Education.html')
display(HTML('Education.html'))


# Sadly this is exactly what we see. Clearly, wealth and education are centered in Manhattan, with about two-thirds having a bachelor's or higher, while it's only about a third in Brooklyn, Staten Island, and Queens. What's worse is when looking at the Bronx, here people with less than high school maintain the highest share of the different education categories. Where this starts to become a big problem is the effect that parents' level of education has on their children. Parents with a lower level of education mean that their children have a much lower likelihood of obtaining a higher level of education [[8]]. This means that boroughs are effectively a positive feedback loop, where well-educated parents produce well-educated children, which gives a higher income making Manhattan even more expensive and so on. Thus boroughs can fuel the discrepancy in NYC. 
# 
# 
# 
# [8]: https://degree.lamar.edu/articles/undergraduate/parents-education-level-and-childrens-success/

# Finally, let's see how the different ethnicities are situated in NYC. Below we plot what percentage of an ethnicity is situated in a given borough.

# In[5]:


## Heatmap for distribution of ethnicities in different boroughs
# Making subplots for each ethnicity
fig = make_subplots(rows=3, cols=2,
                    subplot_titles=list(Ethnicity_map.values()),
                    specs=[[{'type':"mapbox"}, {'type':"mapbox"}],[{'type':"mapbox"}, {'type':"mapbox"}],
                    [{'type':"mapbox", 'colspan':2}, None]],
                    horizontal_spacing= 0.05,
                    vertical_spacing=0.08) 
list_edu = list(Ethnicity_map.keys())
# Subplot of white
i = list_edu[0]
X_grouped = round(data_adult[data_adult['Ethnicity'] == i].groupby(['Boro']).count()/len(data_adult[data_adult['Ethnicity'] == i]),2)*100
X_grouped['BoroName'] = ['Bronx','Brooklyn','Manhattan','Queens','Staten Island']
X_grouped.set_index('BoroName',inplace=True)
att = 'Total_income'
label_text = 'Percentage of ' +  Ethnicity_map[i] + ':'
fig1 = px.choropleth_mapbox(X_grouped[att].reset_index(), geojson=gdf['geometry'], locations=gdf.index, color='Total_income',
                        color_continuous_scale="Viridis",
                        range_color=(X_grouped[att].min(),X_grouped[att].max()),
                        mapbox_style="carto-positron",
                        zoom=8.5, center = {"lat": 40.730610, "lon": -73.935242},
                        opacity=0.5,
                        labels={att: label_text}
                        )
# Subplot of black
i = list_edu[1]
X_grouped = round(data_adult[data_adult['Ethnicity'] == i].groupby(['Boro']).count()/len(data_adult[data_adult['Ethnicity'] == i]),2)*100
X_grouped['BoroName'] = ['Bronx','Brooklyn','Manhattan','Queens','Staten Island']
X_grouped.set_index('BoroName',inplace=True)
att = 'Total_income'
label_text = 'Percentage of ' +  Ethnicity_map[i] + ':'
fig2 = px.choropleth_mapbox(X_grouped[att].reset_index(), geojson=gdf['geometry'], locations=gdf.index, color='Total_income',
                        color_continuous_scale="Viridis",
                        range_color=(X_grouped[att].min(),X_grouped[att].max()),
                        mapbox_style="carto-positron",
                        zoom=8.5, center = {"lat": 40.730610, "lon": -73.935242},
                        opacity=0.5,
                        labels={att:label_text}
                        )
# Subplot of asian
i = list_edu[2]
X_grouped = round(data_adult[data_adult['Ethnicity'] == i].groupby(['Boro']).count()/len(data_adult[data_adult['Ethnicity'] == i]),2)*100
X_grouped['BoroName'] = ['Bronx','Brooklyn','Manhattan','Queens','Staten Island']
X_grouped.set_index('BoroName',inplace=True)
att = 'Total_income'
label_text = 'Percentage of ' +  Ethnicity_map[i] + ':'
fig3 = px.choropleth_mapbox(X_grouped[att].reset_index(), geojson=gdf['geometry'], locations=gdf.index, color='Total_income',
                        color_continuous_scale="Viridis",
                        range_color=(X_grouped[att].min(),X_grouped[att].max()),
                        mapbox_style="carto-positron",
                        zoom=8.5, center = {"lat": 40.730610, "lon": -73.935242},
                        opacity=0.5,
                        labels={att:label_text}
                        )
# Subplot of hispanic
i = list_edu[3]
X_grouped = round(data_adult[data_adult['Ethnicity'] == i].groupby(['Boro']).count()/len(data_adult[data_adult['Ethnicity'] == i]),2)*100
X_grouped['BoroName'] = ['Bronx','Brooklyn','Manhattan','Queens','Staten Island']
X_grouped.set_index('BoroName',inplace=True)
att = 'Total_income'
label_text = 'Percentage of ' +  Ethnicity_map[i] + ':'
fig4 = px.choropleth_mapbox(X_grouped[att].reset_index(), geojson=gdf['geometry'], locations=gdf.index, color='Total_income',
                        color_continuous_scale="Viridis",
                        range_color=(X_grouped[att].min(),X_grouped[att].max()),
                        mapbox_style="carto-positron",
                        zoom=8.5, center = {"lat": 40.730610, "lon": -73.935242},
                        opacity=0.5,
                        labels={att:label_text}
                        )
# Subplot of other
i = list_edu[4]
X_grouped = round(data_adult[data_adult['Ethnicity'] == i].groupby(['Boro']).count()/len(data_adult[data_adult['Ethnicity'] == i]),2)*100
X_grouped['BoroName'] = ['Bronx','Brooklyn','Manhattan','Queens','Staten Island']
X_grouped.set_index('BoroName',inplace=True)
att = 'Total_income'
label_text = 'Percentage of ' +  Ethnicity_map[i] + ':'
fig5 = px.choropleth_mapbox(X_grouped[att].reset_index(), geojson=gdf['geometry'], locations=gdf.index, color='Total_income',
                        color_continuous_scale="Viridis",
                        range_color=(X_grouped[att].min(),X_grouped[att].max()),
                        mapbox_style="carto-positron",
                        zoom=8.5, center = {"lat": 40.730610, "lon": -73.935242},
                        opacity=0.5,
                        labels={att:label_text}
                        )
# Layout and show heatmap
fig.add_trace(fig1['data'][0], row=1, col=1)
fig.add_trace(fig2['data'][0], row=1, col=2)
fig.add_trace(fig3['data'][0], row=2, col=1)
fig.add_trace(fig4['data'][0], row=2, col=2)
fig.add_trace(fig5['data'][0], row=3, col=1)
fig.update_mapboxes(style='carto-positron',
                    zoom=7.5, center = {"lat": 40.730610, "lon": -73.935242})
fig.update_layout(autosize=False, width = 800, height=900, title_text='Percentage of ethnicities in boroughs', title_x=0.5)
fig.layout.coloraxis.colorbar.title = '%'
px_plot(fig, filename = 'Ethnicity_map.html')
display(HTML('Ethnicity_map.html'))


# Here another bleak picture is painted. Manhattan is predominantly white meaning wealth and education are still white, and the Bronx is also mostly not white, thus white is also not poor. Hispanics are mostly in the Bronx which corresponds to the fact that Hispanics are the worst educated Ethnicity. Ending on a high note, we do see that Blacks are well spread out in NYC but Hispanics account for 80% of the population of Bronx. 
