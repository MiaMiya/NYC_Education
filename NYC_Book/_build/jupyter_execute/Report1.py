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


# # Educational segregation in NYC
# 

# "Achieving inclusive and quality education for all reaffirms the belief that education is one of the most powerful and proven vehicles for sustainable development." - [UN](https://www.undp.org/sustainable-development-goals) 
# 
# It should be the goal of any governing organization to ensure high-quality education for all, as its benefits are high and extensive. From the fact that illiteracy means you have a substantially higher likelihood of ending up in jail or on welfare, that illiteracy has a negative impact on discrimination and preventable diseases, or the fact that for every dollar spent on adult illiteracy the ROI (return on investment) is 6.14$ (614%). Another extremely important effect of education is the social network you get, which combats loneliness which in itself has several negative health impacts [[1]].  
# Given that there is no doubt about the importance of education it's important to investigate when the educational system fails and people drop out, and which factors have an impact on the dropout. To investigate this we'll look at poverty data from New York City in 2015, where we focus on education, since this is one of the 17 SDGs, and education is important for developing our world in a sustainable direction. 
# 
# If you want to look behind the scene we have our **explainer notebook** [here](https://nbviewer.org/github/MiaMiya/NYC_Education/blob/main/Explainer_notebook.ipynb) or our github repository [here](https://github.com/MiaMiya/NYC_Education).
# 
# [1]: https://www.hrsa.gov/enews/past-issues/2019/january-17/loneliness-epidemic

# ## An intorduction to the data
# You can download the data from [data.cityofnewyork](https://data.cityofnewyork.us/City-Government/NYCgov-Poverty-Measure-Data-2015-/ahcp-vck8).  
# It contains 69103 participants and 61 columns of which we only use a subset of 11 columns seen below:
# * *Age* of person
# * *Borough* (Bronx, Brooklyn, Manhattan, Queens, Staten Island) 
# * *Disability* 
# * *Level of education* (no high school, high school, some college, bachelor degree or above)
# * *Ethnicity* (white, black, asian hispanic, other)
# * *Language other than english spoken at home* (yes/no) 
# * *Sex* 
# * *Total income* 
#     * *Interest, dividends, and net rental income* 
#     * *Self-employment income* 
#     * *Wages or salary income* 
#     * *Retirement income* 
# 
# We mainly use the people's educational status, since this is what we are interested in investigating. Additionally, we use other above-stated features, to see which have an influence on education. 
# 
# Furthermore, we are only looking at adults (people older than 24 years old) as people younger than 24 have not had a fair opportunity to finish a bachelor's degree. Hence we remove all rows with younger people, this gives us a little less than 50000 participants.  
# 
# The data is generated annually by a research unit in the Mayor's office. It is derived from the American Community Survey Public Use Micro sample for NYC. We will be using the data generated for 2015 as the foundation for this article. 

# ## A look at education in NYC in regards to SDG 4.1 
# 
# The main subject at play is which education people have attained. In [[sdg 4.1]](https://sdgs.un.org/goals/goal4) we see that the goal is for all to have finished a second-level education which corresponds to high school in our case.

# In[4]:


pd.DataFrame(round(data_adult.groupby('EducAttain').count()['SERIALNO']/len(data_adult)*100)).rename(columns = {'SERIALNO':'Percentage'}).rename({1:'Less than High School',2:'High School Degree',3:'Some College',4:'Bachelors Degree or higher'})


# For our dataset, we do see 18% not achieving this goal. We will Further investigate what impact this has.

# ### How education impacts salary 
# 
# An obvious attribute that we would expect education to have an impact on is income. Intuitively, more education leads to better and more well-paid jobs, so let's investigate this claim. We'll do this by looking at the distribution of *total income* in the different education groups

# In[5]:


## Histogram of total income for different education levels
figure(figsize=(15, 10), dpi=80)
i = 1
# For loop over education categories
for dist in sorted(np.unique(data_adult['EducAttain'])): 
    data_boro = data_adult[data_adult['EducAttain'] == dist]
    plt.subplot(3, 2, i)
    sns.histplot(data=data_boro, x="Total_income", bins = bins_freedamn_diaconis(data_boro['Total_income']))
    plt.axvline(data_boro['Total_income'].mean(), color = 'r')
    plt.axvline(data_boro['Total_income'].median(), color = 'darkviolet')
    plt.axvline(data_boro['Total_income'].max(), color = 'g')
    plt.legend(labels=['Average ' + str(round(data_boro['Total_income'].mean())) + 'k','Median ' + str(round(data_boro['Total_income'].median())) + 'k',
     'Max ' + str(round(data_boro['Total_income'].max())) + 'k', EducAttain_map[dist]], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlim(0,data_adult['Total_income'].max()*1.10)
    plt.xlabel('Annual income in thousand dollars')
    plt.ylabel('Count')
    i += 1
# Layout and show plot
plt.suptitle('Distribution of total self made income for the education obtained in thousand dollars pr. year')
plt.tight_layout()
plt.subplots_adjust(top=0.96)
plt.show()


# Interestingly, the individual with the second highest income is (the max line in <font color='green'>green</font>) without a high school diploma ie. more than both people with a high school degree and some college. What this means is just that a high salary/income can be obtained without having any education and not that you can expect a lower maximum income if you have some college education. In fact, what we see is true that you can expect a higher salary the higher your education level. This can be seen in both the average (<font color='red'>red line</font>), and median (<font color='purple'>purple line</font>). Thus it's easy to conclude that education is an effective tool against poverty. However, it's important to note that we know nothing of the jobs that people occupy, so a higher salary does not necessarily mean a job that is a "vehicle for sustainable development" [[UN]](https://www.undp.org/sustainable-development-goals).

# ### Sex a hopeful story
# SDG 4 says: "Ensure inclusive and equitable quality education and promote lifelong learning opportunities for all" [[2]] this unsurprisingly also includes women. Worldwide we know there is a discrepancy between males and females from Hans Rosling's quiz in the opening of his famous book Factfulness: "Worldwide, 30-year-old men have spent 10 years in school, on average. How many years have women of the same age spent in school?" the answer is 9 years [[3]]. This number is of course not US or NYC specific, thus let's take a look at the distribution of sexes for the different achieved educations in NYC: 
# 
# [2]: https://sdgs.un.org/goals/goal4
# [3]: https://factfulnessquiz.com

# In[6]:


# Counting amount og each sex for the different education levels
df_city_health = pd.DataFrame(data_adult.groupby(['EducAttain','SEX'])['SERIALNO'].count()).reset_index()
df_city_health['SEX'].replace([2,1],['Female','Male'],inplace=True)
df_city_health.rename(columns={'SERIALNO': 'Count'}, inplace=True)
df_city_health.loc[df_city_health['SEX'] == 'Male', 'Count'] /= sum(df_city_health.loc[df_city_health['SEX'] == 'Male', 'Count'])
df_city_health.loc[df_city_health['SEX'] == 'Female', 'Count'] /= sum(df_city_health.loc[df_city_health['SEX'] == 'Female', 'Count'])

## Bar plot of amount of each gender for education categories 
figure(figsize=(10, 10), dpi=80)
i = 1
# For loop over education categories
for dist in sorted(set(df_city_health['EducAttain'])): 
    plt.subplot(3, 2, i)
    ax = sns.barplot(x = 'SEX', y = 'Count', data = df_city_health[df_city_health.EducAttain == dist], order = sorted(set(df_city_health.SEX)),        palette="Blues_d")
    plt.legend(labels=[EducAttain_map[dist]])
    plt.xlabel('')
    plt.ylabel('Percentage in sex')
    plt.ylim(0, df_city_health['Count'].max()*1.2)
    i += 1
# Layout and show plot
plt.suptitle('Distribution of sexes for each education level')
plt.tight_layout()
plt.subplots_adjust(top=0.96)
plt.show()


# It's fairly clear that there is no difference in education obtained between the sexes. So nonetheless they are doing pretty good in NYC regarding gender equality in the educational system

# ### Salary and Sex a sad story
# Although the equality of education between the sexes is a good sign, it's an entirely different and alarming story when looking at sex and salary: 
# 

# In[7]:


## Histogram of income for each sex
figure(figsize=(15, 10), dpi=80)
i = 1
# For loop over the sexes
for dist in sorted(np.unique(data_adult['SEX'])): 
    data_boro = data_adult[data_adult['SEX'] == dist]
    plt.subplot(3, 2, i)
    sns.histplot(data=data_boro, x="Total_income", bins = bins_freedamn_diaconis(data_boro['Total_income']))
    plt.axvline(data_boro['Total_income'].mean(), color = 'r')
    plt.axvline(data_boro['Total_income'].median(), color = 'darkviolet')
    plt.axvline(data_boro['Total_income'].max(), color = 'g')
    plt.legend(labels=['Average ' + str(round(data_boro['Total_income'].mean())) + 'k','Median ' + str(round(data_boro['Total_income'].median())) + 'k',
     'Max ' + str(round(data_boro['Total_income'].max())) + 'k'], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlim(data_adult['Total_income'].min(),data_adult['Total_income'].max()*1.10)
    plt.xlabel('Annual income in thousand dollars for ' + SEX_map[dist] + 's')
    plt.ylabel('Count')
    i += 1
# Layout and show plot
plt.suptitle('Distribution of total self made income for the two genders in thousand dollars pr. year')
plt.tight_layout()
plt.subplots_adjust(top=0.96)
plt.show()


# Here we have a fairly big discrepancy as both the <font color='red'>average</font> and <font color='purple'>median</font> is significantly higher for males (about 70%). This is very alarming as it contradicts our previous conclusion: that higher education means a higher salary and thus an effective tool against poverty. The two figures above suggest that although females have an equal amount of education as men, they still have a lower average salary, and thus a higher likelihood of being in poverty. Now you might think that this low salary could be explained by Stay-at-home-moms but there is a difference in salary between men and women even when removing instances of people not earning any money:

# In[8]:


# Removing all people without an annual self made income
data_above = data_adult[data_adult['Total_income'] != 0]

## Histogram of income for each sex with non income people removed 
figure(figsize=(15, 10), dpi=80)
i = 1
# For loop over each sex
for dist in sorted(np.unique(data_adult['SEX'])): 
    data_boro = data_above[data_above['SEX'] == dist]
    plt.subplot(3, 2, i)
    sns.histplot(data=data_boro, x="Total_income", bins = bins_freedamn_diaconis(data_boro['Total_income']))
    plt.axvline(data_boro['Total_income'].mean(), color = 'r')
    plt.axvline(data_boro['Total_income'].median(), color = 'darkviolet')
    plt.axvline(data_boro['Total_income'].max(), color = 'g')
    plt.legend(labels=['Average ' + str(round(data_boro['Total_income'].mean())) + 'k','Median ' + str(round(data_boro['Total_income'].median())) + 'k',
     'Max ' + str(round(data_boro['Total_income'].max())) + 'k'], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlim(data_adult['Total_income'].min(),data_adult['Total_income'].max()*1.10)
    plt.xlabel('Annual income in thousand dollars for ' + SEX_map[dist] + 's')
    plt.ylabel('Count')
    i += 1
# Layout and show plot
plt.suptitle('Distribution of total self made income for the two genders in thousand dollars pr. year, excluding non income people')
plt.tight_layout()
plt.subplots_adjust(top=0.96)
plt.show()


# The difference is still significant (about 35%). And even if the entire difference could be explained by stay-at-home-momes, there is still a question if it should be the case, as this is gender inequality no matter if it is voluntary or not.

# ### The blooming of educated non-white ethnicities
# NYC is a multicultural city with people coming from all ethnicities. Do all of them get the same education opportunities?

# In[9]:


## Counting education levels for different ethnicities
df_city_health = pd.DataFrame(data_adult.groupby(['EducAttain','Ethnicity'])['SERIALNO'].count()).reset_index()
df_city_health.rename(columns={'SERIALNO': 'Count'}, inplace=True)
df_city_health['EducAttain'].replace([1,2,3,4],['Less than High School','High School Degree','Some College','Bachelor\'s Degree or higher'],inplace=True)
for eth in list(Ethnicity_map.keys()): 
    df_city_health.loc[(df_city_health['Ethnicity'] == eth) , 'Count'] = df_city_health.loc[(df_city_health['Ethnicity'] == eth)  , 'Count']/(sum(df_city_health.loc[df_city_health['Ethnicity'] == eth, 'Count']))

## Bar plot of education level for each ethnicity
figure(figsize=(10, 10), dpi=80)
i = 1
# For loop over ethnicities
for dist in sorted(set(df_city_health['Ethnicity'])): 
    plt.subplot(3, 2, i)
    ax = sns.barplot(x = 'EducAttain', y = 'Count', data = df_city_health[df_city_health.Ethnicity == dist],palette="Blues_d")
    plt.xlabel('')
    plt.ylabel('Percentage')
    plt.legend(labels=[Ethnicity_map[dist]])
    plt.ylim(0, 1)
    i += 1
    ax.set_xticklabels(ax.get_xticklabels(),rotation = 30)
# Layout and show plot
plt.suptitle('Distribution of education obtained for the different ethnicities')
plt.tight_layout()
plt.subplots_adjust(top=0.96)
plt.show()


# We do see that white, Asian, and others over 50% have a higher education, whereas the majority of the Hispanics in our dataset have less than high school education which is beneath [SDG 4.1](https://sdgs.un.org/goals/goal4). But could this be a historical issue and no longer be the case? Looking at general education we would expect it to be higher the lower the age (for people older than 24 years old) since the focus on education and resource helping people to get an education has changed dramatically. Additionally, the American society is generally less segregated, especially compared to say the 60s, thus we would expect to see a greater increase in education for all other races.

# In[10]:


### Age, ethnicity and education 
hue_order_edu = list(EducAttain_map.values())
figure(figsize=(15, 12), dpi=80)
i = 1
# For loop over etnicities
for eth in list(Ethnicity_map.keys()):
    plt.subplot(3, 2, i)
    data_temp =  data_adult[data_adult['Ethnicity'] == eth]
    temp_col = data_temp['EducAttain'].map(EducAttain_map)
    data_temp.insert(1, "Education", temp_col, True)
    
    ax = sns.histplot(x = 'AgeGroup',hue = 'Education', data = pd.DataFrame(data_temp), multiple="dodge", shrink=.8,palette="Blues_d", stat='probability',
        hue_order=hue_order_edu)
    plt.title(Ethnicity_map[eth])
    plt.xlabel('Age')
    plt.ylabel('Percentage')
    i += 1
# Layout and show plot
plt.suptitle('Distribution of education obtained divided in age groups, for each ethnicity')
plt.tight_layout()
plt.show()


# The Hispanic race seems to be having the most trouble with obtaining an education even for the younger generation of 20-30, where there are still some age groups where high school education is the most frequent. This may be troublesome for many due to the increasing importance of a college degree steadily increasing [[4]](https://www.ncbi.nlm.nih.gov/books/NBK19909). But alike our theory the younger generation is having a higher education which is a result of the increased focus in education.

# ## The vicious circle in boroughs

# New York City is divided into five different boroughs each with its own flavor [[5]](https://www.britannica.com/place/New-York-City/The-boroughs):  
# The **Bronx** is one of the most prominent centers of urban poverty in the United States.  
# **Brooklyn** collision of old and new   
# **Manhattan** center of NYC and the representative of NYC with central Park, Broadway show, and Times Square  
# **Queens** primary middle-class families and the most ethnically varied of all the boroughs  
# **Staten Island** the most rural part of the city

# Thus it seems fairly intuitive that these boroughs also represent different demographics of the NYC population. To investigate how different the demographics in the different boroughs are, let's start by looking at the average income. 

# In[11]:


## Heatmap of income for each borough
X_grouped = round(data_adult.groupby(['Boro']).median()*1000)
gdf = gpd.read_file('https://raw.githubusercontent.com/dwillis/nyc-maps/master/boroughs.geojson')
gdf.to_crs(epsg=4326, inplace=True)
gdf.set_index('BoroName', inplace=True)
gdf['BoroCode'] = [5,4,2,3,1]
gdf.sort_index(inplace=True)
X_grouped['BoroName'] = ['Bronx','Brooklyn','Manhattan','Queens','Staten Island']
X_grouped.set_index('BoroName',inplace=True)
att = 'Total_income'
fig = px.choropleth_mapbox(X_grouped[att].reset_index(), geojson=gdf['geometry'], locations=gdf.index, color='Total_income',
                           color_continuous_scale="Viridis",
                           range_color=(X_grouped[att].min(),X_grouped[att].max()),
                           mapbox_style="carto-positron",
                           zoom=8.5, center = {"lat": 40.730610, "lon": -73.935242},
                           opacity=0.5,
                           labels={att: 'Median total income in borough'}
                          )
# Show plot
px_plot(fig, filename = 'figure_1.html')
display(HTML('figure_1.html'))


# Clearly, there is a big difference between living in Manhattan and Bronx, the average income in Manhattan is over three times that of the Bronx, even though they are two neighboring boroughs. Where Brooklyn, Staten Island, and Queens are much more similar but still far behind Manhattan. However, Manhattan is as mentioned home to all the biggest and most shining stars of capitalism like Wall Street, Trump Tower, Empire State Building, etc. which not only means insanely high property values (it is 3 times the average [[6]] of NYC [[7]]), but also an aggregation of wealth. 
# 
# 
# 
# [6]: https://www.redfin.com/city/35948/NY/Manhattan/housing-market
# [7]: https://www.noradarealestate.com/blog/new-york-real-estate-market/

# As we've mentioned previously there is a correlation between a high income and high level of education. Thus the heatmap above would seem to indicate that a lot of highly educated people will be centered in Manhattan, with Bronx severely lacking behind. Let's see this in heatmaps of percentege of people with a certain level of education by boroughs. 
