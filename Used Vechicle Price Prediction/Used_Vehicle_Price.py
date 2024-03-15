# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 21:08:25 2023

@author: Hemant
"""

import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_theme(color_codes=True)

df = pd.read_csv('./assets/originalData.csv')
df.columns
# =============================================================================
# Data Cleanning
# =============================================================================
df.drop(columns=['Unnamed: 0'], inplace=True)
df.shape
df.info()
# Cleaing the space in columns name 
for i in range(len(df.columns)):
    df.rename(columns={df.columns[i]:df.columns[i].strip()}, 
              inplace=True)
    
# Remove Rows with missing valie at 'City' and 'Highwat' columns
df.dropna(subset=['City', 'Highway'], inplace=True)
# Removing the per 100km from the City and Highway
df[ 'City'] = df['City'].str.split('L', n=1, expand = True)[0]
df[ 'Highway'] = df['Highway'].str.split('L', n=1, expand = True)[0]
# Remove the nmo-numeric character from city and highway
df['City'] = df['City'].str.replace('[^0-9\.]', '').astype('float')
df['Highway'] = df['Highway'].str.replace('[^0-9\.]', '').astype('float')
# Remove the rows having missing avlues in Kilometers and extract all the 
df.dropna(subset=['Kilometres'], inplace=True)
df['Kilometres'] = df['Kilometres'].str.extract(r'([0-9]*)').astype(int)

# Now Iterate Over columns with object datatypes
# Find the nunique valus for that columns and plot it
nunique_object = df.select_dtypes(include='object').nunique()
plt.figure(figsize=(20,9))
ax = nunique_object.plot(kind='bar', log=True)
# adding the the numeric text into the bar 
for i, v in enumerate(nunique_object):
    ax.text(i, v/2, str(v), ha='center', va='center', color='black', fontweight='bold', fontsize=20)
plt.ylabel('Number Of Unique Values')        
plt.ylabel('Cars Fetures')      
plt.title('Fetures/UniqueValues')  
plt.show()    

# Replace the Model Value by 'Other Model' and plot 
df['Model'] = df['Model'].str.replace(r'^(?!MDX|TSX|Grand|Civic|RDX|ILX|TLX).*$', 'Other Model')
plt.figure(figsize=(10,5))
ax = df['Model'].value_counts().plot(kind='bar', log=True)
for i, v in enumerate(df['Model'].value_counts()):
    ax.text(i, v/2, str(v), ha='center', va='center', color='black', fontweight='bold', fontsize=10)
plt.ylabel('Value Counts')
plt.xlabel('Model Name')
plt.show()

# Droping unrequired Columns
df.drop(columns=['Exterior Colour', 'Interior Colour'], inplace=True)
df.shape

# Now Segmenta the brand into Luxury, Mainstream, Sports, Value, and Other
df['Make'].unique()
# Define the the lists for segmentation 
Luxury = ['Acura', 'Alfa Romeo', 'Audi', 'Bentley', 
          'BMW', 'Cadillac', 'Genesis', 'Infiniti', 
          'Jaguar', 'Lamborghini', 'Land Rover', 
          'Lexus', 'Lincoln', 'Maserati', 'McLaren', 
          'Mercedes-Benz', 'Porsche', 'Rolls-Royce', 
          'Tesla']
Mainstream = ['Buick', 'Chevrolet', 'Chrysler', 
              'Dodge', 'Ford', 'GMC', 'Jeep', 'Ram']
Sports = ['Ferrari', 'Lotus']
Value = ['Honda', 'Hyundai', 'Kia', 'Mazda', 
         'Mitsubishi', 'Nissan', 'Subaru', 'Toyota', 
         'Volkswagen']
def segment_make(make):
    if make in Luxury:
        return 'Luxury'
    elif make in Mainstream:
        return 'Mainstream'
    elif make in Sports:
        return 'Sports'
    elif make in Value:
        return 'Value'
    else:
        return 'Other'
df['Make'] = df['Make'].apply(segment_make)
# Now We are going to plot the Values 
plt.figure(figsize = (10, 5))
ax = df['Make'].value_counts().plot(kind='bar', log=True)
for i, v in enumerate(df['Make'].value_counts()):
    ax.text(i, v/2, str(v), ha='center', va='center', color='black', fontweight='bold', fontsize=10)
plt.ylabel('Value Counts')
plt.xlabel('Make Name')
plt.show()

# Segment the Body type 
df['Body Type'].unique()
# Define the list For segmentation
SUV = ['SUV']
Sedan = ['Sedan', 'Coupe', 'Convertible']
Hatchback = ['Hatchback']
Wagon = ['Wagon', 'Station Wagon']
Truck = ['Truck', 'Truck Extended Cab','Extended Cab', 
         'Crew Cab', 'Regular Cab', 'Truck Crew Cab', 'Super Cab',
         'Quad Cab', 'Truck Super Cab','Truck Double Cab', 'Truck King Cab',
         'Truck Long Crew Cab']
Van = ['Van Regular', 'Van Extended']
Minivan = ['Minivan']
Roadster = ['Roadster']
Cabriolet = ['Cabriolet']
Super_Crew = ['Super Crew']
Compact = ['Compact']
# Create The Dictionary to map each body type to its corresponding segment
body_type_segment = dict()
for body_type in df['Body Type'].unique():
    if body_type in SUV:
        body_type_segment[body_type] = 'SUV'
    elif body_type in Sedan:
        body_type_segment[body_type] = 'Sedan'
    elif body_type in Hatchback:
        body_type_segment[body_type] = 'Hatchback'
    elif body_type in Wagon:
        body_type_segment[body_type] = 'Wagon'
    elif body_type in Truck:
        body_type_segment[body_type] = 'Truck'
    elif body_type in Van:
        body_type_segment[body_type] = 'Van'
    elif body_type in Minivan:
        body_type_segment[body_type] = 'Minivan'
    elif body_type in Roadster:
        body_type_segment[body_type] = 'Roadster'
    elif body_type in Cabriolet:
        body_type_segment[body_type] = 'Cabriolet'
    elif body_type in Super_Crew:
        body_type_segment[body_type] = 'Super Crew'
    elif body_type in Compact:
        body_type_segment[body_type] = 'Compact'
    else: 
        body_type_segment[body_type] = 'Other'
# Map the function body type and segment
df['Body Type'] = df['Body Type'].map(body_type_segment)
# Ploting the countplot
plt.figure(figsize=(10, 9))
ax = df['Body Type'].value_counts().plot(kind='bar', log=True)
for i, v in enumerate(df['Body Type'].value_counts()):
    ax.text(i, v/2, str(v), ha='center', va='center', fontweight='bold', fontsize=9, rotation=90)
plt.ylabel('Value Counts')
plt.xlabel('Body Type')
plt.show()

# Segment Transmission
df['Transmission'].unique()
def segement_transmission(transmission):
   if transmission in ['Automatic', 'CVT', '1 Speed Automatic']:
       return 'Automatic'
   elif transmission in ['6 Speed Manual', '5 Speed Manual', '7 Speed Manual']:
       return 'Manual'
   elif transmission in ['9 Speed Automatic', '10 Speed Automatic', '8 Speed Automatic', '7 Speed Automatic', '5 Speed Automatic', '4 Speed Automatic']:
       return 'Traditional Automatic'
   elif transmission in ['8 Speed Automatic with auto-shift', '6 Speed Automatic with auto-shift', '7 Speed Automatic with auto-shift', '5 Speed Automatic with auto-shift']:
       return 'Automated Manual'
   elif transmission == 'Sequential':
       return 'Semi-Automatic'
   elif transmission == 'F1 Transmission':
       return 'Automated Single-Clutch'
   else:
       return 'Unknown'
df['Transmission'] = df['Transmission'].apply(segement_transmission)
# Plotiing the data 
plt.figure(figsize=(10, 9))
ax = df['Transmission'].value_counts().plot(kind='bar', log= True)
for i, v in enumerate(df['Transmission'].value_counts()):
    ax.text(i, v+0.5, str(v), ha='center', va='center', fontweight='bold', fontsize=15)
plt.xlabel('Value Count')
plt.xlabel('Transmission')
plt.show()
# Droping the transmission columns becasue it has a lot of unknowns transmission type of the vechical
df.drop(columns= ['Transmission'], inplace=True)
df.shape

# Drop the engine columnns 
df.drop(columns=['Engine'], inplace=True)
df.shape
df.head()

# =============================================================================
# Exploratory Data Analysis
# =============================================================================
# Ploting the unique number of categorical values col having 
obj_uni = df.select_dtypes(include='object').nunique()
print(obj_uni)
plt.figure(figsize=(10, 6))
obj_uni.plot(kind='bar')

# Categorical Vlaue Vs Price Plot
plt.figure(figsize=(20, 20))
for i, var in enumerate(df.select_dtypes(include='object')):
    plt.subplot(3, 2, i+1)
    sns.barplot(x=var, y='Price', data=df)  
    plt.xticks(rotation = 90)   
plt.tight_layout()    
plt.show()












