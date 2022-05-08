#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from urllib.request import urlopen
import json
import geopandas as gpd


# ## Import data and preprocess

# In[ ]:


data = pd.read_csv("NYCgov_Poverty_Measure_Data__2015_.csv")


# In[ ]:


features = ['SERIALNO', 'SPORDER', 'AGEP', 'CIT', 'REL', 'SCH',
       'SCHG', 'SCHL', 'SEX', 'ESR', 'LANX', 'ENG', 'MSP',
       'WKHP', 'DIS', 'JWTR', 'NP', 'TEN', 'HHT', 'AgeCateg', 'Boro',
       'CitizenStatus', 'EducAttain', 'Ethnicity', 'FamType_PU', 'FTPTWork', 
       'INTP_adj', 'MRGP_adj', 'NYCgov_Income', 'NYCgov_Pov_Stat', 'NYCgov_REL',
       'NYCgov_Threshold', 'Off_Pov_Stat', 'Off_Threshold', 'OI_adj', 'PA_adj', 
       'Povunit_ID', 'Povunit_Rel', 'PreTaxIncome_PU', 'RETP_adj', 'RNTP_adj', 
       'SEMP_adj', 'SSIP_adj', 'SSP_adj', 'TotalWorkHrs_PU', 'WAGP_adj']

#Recode = code in dictionary
# CIT: Citenzenship
# REL: is relationship ie. Daughter, Son, etc. is ACS code ()
# SCH, SCHG: (SCHG is ACS code) for educaiton
# SCHL: Education attainment ACS code
# ESR: Employement status (code in dictionary file)
# LANX: language other than language spoken 
# ENG: ability to speak english
# MSP: Married or not (code in dictionary file)
# MAR: Marital status 
# WKHP: huors work per week
# DIS: disability (Recode)
# JWTR: transportation to work (ACS)
# NP: number of people in household 
# TEN: Housing tenure
# FamType_PT: PovertyUnit familytype (umiddelbart fjerne)
# FTPTWork: work experience (recode)
# INTP_adj: Income adjusted
# MRGP_adj: Morgage amount adjusted
# SEMP_adj: self employed
# SSIP_adh: supplementary income 
# SSP_adj: social socurity income (people who are disabled)
# WAGP_adj: Wages


# ## Visulization 

# * Number of healty tree in each district 
# * Probablity of healthy tree in each district 
# * histogram of diameter
# * histogram of depth
# * plot of location for trees (heatmap)

# In[ ]:


X = data[features]


# In[ ]:


i = 1
plt.figure(figsize=(25, 25))
for feature in features[1:]:
    plt.subplot(4, 4, i)
    if feature == 'spc_common':
        g = sns.scatterplot(x='spc_common',y='tree_dbh',hue='health',data=X)
        g.set_xticks(np.arange(0,name_fact[0].max(),20)) # <--- set the ticks first
        g.set_xticklabels(np.arange(0,name_fact[0].max(),20))
    else:
        sns.scatterplot(x=feature,y='tree_dbh',hue='health',data=X)
    i += 1


# In[ ]:


X_grouped = X.groupby(['Boro']).median()


# In[ ]:


gdf = gpd.read_file('https://raw.githubusercontent.com/dwillis/nyc-maps/master/boroughs.geojson')
gdf.to_crs(epsg=4326, inplace=True)
gdf.set_index('BoroName', inplace=True)
gdf['BoroCode'] = [5,4,2,3,1]
gdf.sort_index(inplace=True)


# In[ ]:


X_grouped['BoroName'] = ['Bronx','Brooklyn','Manhattan','Queens','Staten Island']
X_grouped.set_index('BoroName',inplace=True)


# In[ ]:


att = 'PreTaxIncome_PU'


# In[ ]:


fig = px.choropleth_mapbox(X_grouped[att], geojson=gdf['geometry'], locations=gdf.index, color=X_grouped[att],
                           color_continuous_scale="Viridis",
                           range_color=(X_grouped[att].min(),X_grouped[att].max()),
                           mapbox_style="carto-positron",
                           zoom=8.9, center = {"lat": 40.730610, "lon": -73.935242},
                           opacity=0.5,
                           labels={att:att}
                          )
fig.update_layout(margin={"r":300,"t":100,"l":200,"b":0})
fig.show("notebook")


# In[ ]:


X_grouped[[att,'EducAttain']]


# ## attributes in ML model
# NP: Number of people in house hold  
# Race  
# Sex  
# Boro  
# Age  
# LANX: language other than language spoken   
# DIS: disability (Recode)  

# In[ ]:




