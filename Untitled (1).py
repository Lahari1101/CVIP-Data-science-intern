#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import warnings
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import plotly.offline as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


# In[2]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[4]:


# Let's import to our data and check the basics.
terrorism = pd.read_csv('globalterrorismdb_0718dist.csv',encoding='ISO-8859-1')


# In[5]:


terrorism.head()


# In[8]:


terrorism.columns


# In[9]:


terrorism.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','provstate':'state',
                       'region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed',
                       'nwound':'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type',
                       'weaptype1_txt':'Weapon_type','motive':'Motive'},inplace=True)


# In[11]:


terrorism=terrorism[['Year','Month','Day','Country','state','Region','city','latitude','longitude','AttackType','Killed',
               'Wounded','Target','Summary','Group','Target_type','Weapon_type','Motive']]


# In[12]:


terrorism.isnull().sum()


# In[13]:


terrorism.info()


# ### Destructive 

# In[15]:


print("Country with the most attacks:",terrorism['Country'].value_counts().idxmax())
print("City with the most attacks:",terrorism['city'].value_counts().index[1]) #as first entry is 'unknown'
print("Region with the most attacks:",terrorism['Region'].value_counts().idxmax())
print("Year with the most attacks:",terrorism['Year'].value_counts().idxmax())
print("Month with the most attacks:",terrorism['Month'].value_counts().idxmax())
print("Group with the most attacks:",terrorism['Group'].value_counts().index[1])
print("Most Attack Types:",terrorism['AttackType'].value_counts().idxmax())


# In[19]:


from wordcloud import WordCloud
from scipy import signal
cities = terrorism.state.dropna(False)
plt.subplots(figsize=(10,10))
wordcloud = WordCloud(background_color = 'white',
                     width = 512,
                     height = 384).generate(' '.join(cities))
plt.axis('off')
plt.imshow(wordcloud)
plt.show()


# In[17]:


pip install wordcloud


# In[20]:


terrorism['Year'].value_counts(dropna = False).sort_index()


# ## Data visualization

# ### no of terrorists activities each year

# In[21]:


x_year = terrorism['Year'].unique()
y_count_years = terrorism['Year'].value_counts(dropna = False).sort_index()
plt.figure(figsize = (18,10))
sns.barplot(x = x_year,
           y = y_count_years,
           palette = 'rocket')
plt.xticks(rotation = 45)
plt.xlabel('Attack Year')
plt.ylabel('Number of Attacks each year')
plt.title('Attack_of_Years')
plt.show()


# In[26]:


plt.subplots(figsize=(15,6))
sns.countplot('Year',data=terrorism,palette='RdYlGn_r',edgecolor=sns.color_palette("YlOrBr", 10))
plt.xticks(rotation=45)
plt.title('Number Of Terrorist Activities Each Year')
plt.show()


# In[24]:


def color_box(color, *args, **kwargs):
    painter.select_color(color)
    painter.draw_box(*args, **kwargs)


# In[28]:


pd.crosstab(terrorism.Year, terrorism.Region).plot(kind='area',figsize=(15,6))
plt.title('Terrorist Activities by Region in each Year')
plt.ylabel('Number of Attacks')
plt.show()


# In[29]:


terrorism['Wounded'] = terrorism['Wounded'].fillna(0).astype(int)
terrorism['Killed'] = terrorism['Killed'].fillna(0).astype(int)
terrorism['casualities'] = terrorism['Killed'] + terrorism['Wounded']


# Values are sorted by the top 40 worst terror attacks as to keep the heatmap simple and easy to visualize

# In[30]:


terrorism1 = terrorism.sort_values(by='casualities',ascending=False)[:40]


# In[34]:


heat=terrorism1.pivot_table(index='Country',columns='Year',values='casualities')
heat.fillna(0,inplace=True)


# In[35]:


heat.head()


# In[36]:


import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
colorscale = [[0, '#edf8fb'], [.3, '#00BFFF'],  [.6, '#8856a7'],  [1, '#810f7c']]
heatmap = go.Heatmap(z=heat.values, x=heat.columns, y=heat.index, colorscale=colorscale)
data = [heatmap]
layout = go.Layout(
    title='Top 40 Worst Terror Attacks in History from 1982 to 2016',
    xaxis = dict(ticks='', nticks=20),
    yaxis = dict(ticks='')
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='heatmap',show_link=False)


# In[38]:


terrorism.Country.value_counts()[:15]


# ### Top countries effected by terror attacks

# In[40]:


plt.subplots(figsize=(15,6))
sns.barplot(terrorism['Country'].value_counts()[:15].values,palette='Blues_d')
plt.title('Top Countries Affected')
plt.xlabel('Countries')
plt.ylabel('Count')
plt.xticks(rotation= 90)
plt.show()


# # ANALYSIS ON CUSTOMIZED DATA
# ## Terrorist Attacks of a Particular year and their Locations
# #### Let's look at the terrorist acts in the world over a certain year.

# In[44]:


import folium
from folium.plugins import MarkerCluster 
filterYear = terrorism['Year'] == 1970


# In[42]:


pip install folium


# In[46]:


filterData = terrorism[filterYear] # filter data
# filterData.info()
reqFilterData = filterData.loc[:,'city':'longitude'] #We are getting the required fields
reqFilterData = reqFilterData.dropna() # drop NaN values in latitude and longitude
reqFilterDataList = reqFilterData.values.tolist()
# reqFilterDataList


# In[47]:


map = folium.Map(location = [0, 30], tiles='CartoDB positron', zoom_start=2)
# clustered marker
markerCluster = folium.plugins.MarkerCluster().add_to(map)
for point in range(0, len(reqFilterDataList)):
    folium.Marker(location=[reqFilterDataList[point][1],reqFilterDataList[point][2]],
                  popup = reqFilterDataList[point][0]).add_to(markerCluster)
map


# In[48]:


terrorism.Group.value_counts()[1:15]


# In[50]:


test = terrorism[terrorism.Group.isin(['Shining Path (SL)','Taliban','Islamic State of Iraq and the Levant (ISIL)'])]
test.Country.unique()


# In[51]:


terrorism_df_group = terrorism.dropna(subset=['latitude','longitude'])
terrorism_df_group = terrorism_df_group.drop_duplicates(subset=['Country','Group'])
terrorismist_groups = terrorism.Group.value_counts()[1:8].index.tolist()
terrorism_df_group = terrorism_df_group.loc[terrorism_df_group.Group.isin(terrorismist_groups)]
print(terrorism_df_group.Group.unique())


# In[53]:


map = folium.Map(location=[20, 0], tiles="CartoDB positron", zoom_start=2)
markerCluster = folium.plugins.MarkerCluster().add_to(map)
for i in range(0,len(terrorism_df_group)):
    folium.Marker([terrorism_df_group.iloc[i]['latitude'],terrorism_df_group.iloc[i]['longitude']], 
                  popup='Group:{}<br>Country:{}'.format(terrorism_df_group.iloc[i]['Group'], 
                  terrorism_df_group.iloc[i]['Country'])).add_to(map)
map


# In[57]:


m1 = folium.Map(location=[20, 0], tiles="CartoDB positron", zoom_start=2)
marker_cluster = MarkerCluster(
    name='clustered icons',
    overlay=True,
    control=False,
    icon_create_function=None
)
for i in range(0,len(terrorism_df_group)):
    marker=folium.Marker([terrorism_df_group.iloc[i]['latitude'],terrorism_df_group.iloc[i]['longitude']]) 
    popup='Group:{}<br>Country:{}'.format(terrorism_df_group.iloc[i]['Group'],
                                          terrorism_df_group.iloc[i]['Country'])
    folium.Popup(popup).add_to(marker)
    marker_cluster.add_child(marker)
marker_cluster.add_to(m1)
folium.TileLayer('openstreetmap').add_to(m1)
#folium.TileLayer('Mapbox Bright').add_to(m1)
folium.TileLayer('cartodbdark_matter').add_to(m1)
folium.TileLayer('stamentoner').add_to(m1)
folium.LayerControl().add_to(m1)

m1


# In[58]:


terrorism.head()


# In[60]:


# Total Number of people killed in terror attack
killData = terrorism.loc[:,'Killed']
print('Number of people killed by terror attack:', int(sum(killData.dropna())))# drop the NaN values


# In[63]:


# Let's look at what types of attacks these deaths were made of.
attackData = terrorism.loc[:,'AttackType']


# In[64]:


# attackData
typeKillData = pd.concat([attackData, killData], axis=1)
typeKillData.head()


# In[65]:


typeKillFormatData = typeKillData.pivot_table(columns='AttackType', values='Killed', aggfunc='sum')
typeKillFormatData


# In[66]:


typeKillFormatData.info()


# In[71]:


'''labels = typeKillFormatData.columns.tolist() # convert line to list
transpoze = typeKillFormatData.T # transpoze
values = transpoze.values.tolist()
fig, ax = plt.subplots(figsize=(20, 20), subplot_kw=dict(aspect="equal"))
plt.pie(values, startangle=90, autopct='%.2f%%')
plt.title('Types of terrorist attacks that cause deaths')
plt.legend(labels, loc='upper right', bbox_to_anchor = (1.3, 0.9), fontsize=15) # location legend
plt.show()'''


# In[69]:


#Number of Killed in Terrorist Attacks by Countries
countryData = terrorism.loc[:,'Country']
# countyData
countryKillData = pd.concat([countryData, killData], axis=1)
countryKillFormatData = countryKillData.pivot_table(columns='Country', values='Killed', aggfunc='sum')
countryKillFormatData


# In[72]:


fig_size = plt.rcParams["figure.figsize"]
fig_size[0]=25
fig_size[1]=25
plt.rcParams["figure.figsize"] = fig_size


# In[73]:


labels = countryKillFormatData.columns.tolist()
labels = labels[:50] #50 bar provides nice view
index = np.arange(len(labels))
transpoze = countryKillFormatData.T
values = transpoze.values.tolist()
values = values[:50]
values = [int(i[0]) for i in values] # convert float to int
colors = ['red', 'green', 'blue', 'purple', 'yellow', 'brown', 'black', 'gray', 'magenta', 'orange'] # color list for bar chart bar color 
fig, ax = plt.subplots(1, 1)
ax.yaxis.grid(True)
fig_size = plt.rcParams["figure.figsize"]
fig_size[0]=25
fig_size[1]=25
plt.rcParams["figure.figsize"] = fig_size
plt.bar(index, values, color = colors, width = 0.9)
plt.ylabel('Killed People', fontsize=20)
plt.xlabel('Countries', fontsize = 20)
plt.xticks(index, labels, fontsize=18, rotation=90)
plt.title('Number of people killed by countries', fontsize = 20)
# print(fig_size)
plt.show()


# In[74]:


labels = countryKillFormatData.columns.tolist()
labels = labels[50:101]
index = np.arange(len(labels))
transpoze = countryKillFormatData.T
values = transpoze.values.tolist()
values = values[50:101]
values = [int(i[0]) for i in values]
colors = ['red', 'green', 'blue', 'purple', 'yellow', 'brown', 'black', 'gray', 'magenta', 'orange']
fig, ax = plt.subplots(1, 1)
ax.yaxis.grid(True)
fig_size = plt.rcParams["figure.figsize"]
fig_size[0]=20
fig_size[1]=20
plt.rcParams["figure.figsize"] = fig_size
plt.bar(index, values, color = colors, width = 0.9)
plt.ylabel('Killed People', fontsize=20)
plt.xlabel('Countries', fontsize = 20)
plt.xticks(index, labels, fontsize=18, rotation=90)
plt.title('Number of people killed by countries', fontsize = 20)
plt.show()


# In[75]:


labels = countryKillFormatData.columns.tolist()
labels = labels[152:206]
index = np.arange(len(labels))
transpoze = countryKillFormatData.T
values = transpoze.values.tolist()
values = values[152:206]
values = [int(i[0]) for i in values]
colors = ['red', 'green', 'blue', 'purple', 'yellow', 'brown', 'black', 'gray', 'magenta', 'orange']
fig, ax = plt.subplots(1, 1)
ax.yaxis.grid(True)
fig_size = plt.rcParams["figure.figsize"]
fig_size[0]=25
fig_size[1]=25
plt.rcParams["figure.figsize"] = fig_size
plt.bar(index, values, color = colors, width = 0.9)
plt.ylabel('Killed People', fontsize=20)
plt.xlabel('Countries', fontsize = 20)
plt.xticks(index, labels, fontsize=18, rotation=90)
plt.title('Number of people killed by countries', fontsize = 20)
plt.show()


# Terrorist acts in the Middle East and northern Africa have been seen to have fatal consequences. The Middle East and North Africa are seen to be the places of serious terrorist attacks. In addition, even though there is a perception that Muslims are supporters of terrorism, Muslims are the people who are most damaged by terrorist attacks. If you look at the graphics, it appears that Iraq, Afghanistan and Pakistan are the most damaged countries. All of these countries are Muslim countries.

# In[ ]:




