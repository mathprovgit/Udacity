#!/usr/bin/env python
# coding: utf-8

# # Analysing Titanic passenger information

# ## Intro do Data Analysis - Lesson4

# ### interesting links
#     - about the data:
#         http://blog.kaggle.com/2016/06/29/communicating-data-science-a-guide-to-presenting-your-work/
#         
#     - about the titanic:
#         https://www.encyclopedia-titanica.org/titanic-passenger-list/
#         https://en.wikipedia.org/wiki/RMS_Titanic
#     
#     - about report of data analysis:
#         https://career-resource-center.udacity.com/portfolio/data-science-reports
#         http://blog.kaggle.com/2016/06/29/communicating-data-science-a-guide-to-presenting-your-work/
#     
#     - about the df.plot():
#         http://queirozf.com/entries/pandas-dataframe-plot-examples-with-matplotlib-pyplot
#         
#       
#         

# ### comments-ideas:
#     number of passengers in the csv file 891 but in titanic 1350 (vary depending of the sources)
#     
#     

# ### Problematique : Question about the data
#     1 What is the passenger surviving rate ? (new technology safe ?)
#     2 Statistics about passenger age? (distribution)
#     3 Are women more likly to survive than men ? (titanic the movie)
#     4 Are rich more likly to survive ? (first class to rest and second class to third)
#     5 What are the most impact variables on the survivinge rate of the titanic passengers? 
#     6 Anything else ?
#     

# #### import requiered packages

# In[518]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math


# #### Open csv file and check some values

# In[495]:


file_name='titanic_data.csv'
full_path=file_name
titanic_df=pd.read_csv(full_path)
num_row,num_col=titanic_df.shape
print('there is {} rows and {} cols in this df'.format(num_row,num_col))
titanic_df.head()


# #### split names
#     Name -> Name, FName and MrsName

# In[496]:


name_df=titanic_df.Name.str.replace('(',',').str.replace(')','').str.split(',',expand=True)
titanic_df['Name']=name_df[0]
titanic_df['Fname']=name_df[1]
titanic_df['Mname']=name_df[2]


# In[497]:


titanic_df[titanic_df['Name']=='Futrelle']


# In[498]:


titanic_df[titanic_df['Name']=='Palsson']


# #### 1: Surviving rate on the RMS Titanic

# In[499]:


#add a dead column
#titanic_df['Dead']=(1+titanic_df.Survived)%2

#calculate the Surviving_rate and the deat rate
survivors_count=titanic_df[titanic_df.Survived==1]['PassengerId'].count()
dead_count=titanic_df[titanic_df.Survived==0]['PassengerId'].count()
passengers_count=titanic_df['PassengerId'].count()
Surviving_rate=round(survivors_count*100/passengers_count,1)
death_rate=round(dead_count*100/passengers_count,1)

#display result
print("The surving rate on the RMS Titanic is about {}% -> {} survivors / {} passengers".format(Surviving_rate,survivors_count,passengers_count))
print("The death rate on the RMS Titanic is about {}%-> {} survivors / {} passengers".format(death_rate,dead_count,passengers_count))


# #### 2: Age distribution on the Titanic

# In[500]:


# titanic_df.Age.max()
# fill nan value with 99 (no trouble since max age =80) 
titanic_df.Age.fillna(99,inplace=True)

# class age in groups of 10 years
titanic_df['Age_class']=((titanic_df.Age*0.1).apply(int)*10).apply(str)+' - '+(((titanic_df.Age*0.1).apply(int)+1)*10).apply(str)

titanic_df_age_group=(titanic_df.groupby(['Survived','Age_class'],group_keys=False,as_index=False).size().unstack()).fillna(0).T
titanic_df_age_group.rename(index={'90 - 100': 'NaN'},inplace=True)
#titanic_df_age_group.loc['total'] = [len(titanic_df[titanic_df.Survived==0]),len(titanic_df[titanic_df.Survived==1])]
titanic_df_age_group.plot(kind='bar',stacked=True
                          ,grid=True,title='Distribution of Age of Titanic Passengers (stacked)',figsize=(6,6),width=0.7)
plt.ylabel('Passenger Count')
plt.show()

#simple solution
#titanic_df[['Age']].plot(kind='hist',bins=[0,10,20,30,40,50,60,70,80,90],rwidth=0.8)


# In[501]:


titanic_df_age_group.loc['Titanic']=Surviving_rate
titanic_df_age_group['Surviving_rate']=round(titanic_df_age_group[1]*100/(titanic_df_age_group[0]+titanic_df_age_group[1]),1)
titanic_df_age_group['Surviving_rate'].plot(kind='bar', title='surviving rate per age group')
plt.ylabel('%')
plt.show()


# In[502]:


def age_to_int(Age):
    if Age<=18:
        return '0_Child'
    else:
        return '1_Adult'
    
titanic_df['Age_code']=titanic_df.Age.apply(age_to_int)
titanic_df_age_group=(titanic_df.groupby(['Survived','Age_code'],group_keys=False,as_index=False).size().unstack()).fillna(0).T
titanic_df_age_group.plot(kind='bar',stacked=True
                          ,grid=True,title='Distribution of Age of Titanic Passengers (stacked)',figsize=(6,6),width=0.7)
plt.ylabel('Passenger Count')
plt.show()


# In[503]:


titanic_df_age_group.loc['Titanic']=Surviving_rate
titanic_df_age_group['Surviving_rate']=round(titanic_df_age_group[1]*100/(titanic_df_age_group[0]+titanic_df_age_group[1]),1)
titanic_df_age_group['Surviving_rate'].plot(kind='bar', title='surviving rate per age group')
plt.ylabel('%')
plt.show()


# #### 3: distribution of gender of the Titanic passenger

# In[504]:


titanic_df_gender=(titanic_df.groupby(['Survived','Sex'],group_keys=False,as_index=False).size().unstack()).T
titanic_df_gender.plot(kind='bar',grid=True,stacked=True,width=0.7,figsize=(6,6)
                       ,title='Gender Distribution of Titanic Passengers (stacked)')
plt.ylabel('Passenger Count')
plt.xlabel('Gender')
plt.show()


# In[505]:


titanic_df_gender.loc['Titanic']=Surviving_rate
titanic_df_gender['Surviving_rate']=round(titanic_df_gender[1]*100/(titanic_df_gender[0]+titanic_df_gender[1]),1)
titanic_df_gender['Surviving_rate'].plot(kind='bar', title='surviving rate per Sex')
plt.ylabel('%')
plt.show()


# #### 4: PClass distribution of the Titanic passenger

# In[506]:


titanic_df_Pclass=(titanic_df.groupby(['Survived','Pclass'],group_keys=False,as_index=False).size().unstack()).T
titanic_df_Pclass.plot(kind='bar',stacked=True
                        ,grid=True,title='PClass Distribution of Titanic Passengers (stacked)',figsize=(6,6),width=0.7)
plt.ylabel('Passenger Count')
plt.show()


# In[507]:


titanic_df_Pclass.loc['Titanic']=Surviving_rate
titanic_df_Pclass['Surviving_rate']=round(titanic_df_Pclass[1]*100/(titanic_df_Pclass[0]+titanic_df_Pclass[1]),1)
titanic_df_Pclass['Surviving_rate'].plot(kind='bar', title='surviving rate per Pclass')
plt.ylabel('%')
plt.show()


# #### 5: PClass, Gender, Age distribution of the Titanic passenger

# In[508]:


titanic_df_Surv_Sex_Pclass=(titanic_df.groupby(['Sex','Age_code','Pclass','Survived'],as_index=False).size().unstack().fillna(0))

#titanic_df_Surv_Sex_Pclass.loc['total'] = list(titanic_df_Surv_Sex_Pclass.sum().values)
titanic_df_Surv_Sex_Pclass.plot(kind='bar',stacked=True
                        ,grid=True,title='PClass and Gender Distribution of Titanic Passengers (stacked)',figsize=(6,6),width=0.7)
plt.ylabel('Passenger Count')
plt.xlabel('Gender, Pclass')
plt.show()
#titanic_df_Surv_Sex_Pclass


# In[509]:


titanic_df_Surv_Sex_Pclass.loc['Titanic']=Surviving_rate
titanic_df_Surv_Sex_Pclass['Surviving_rate']=round(titanic_df_Surv_Sex_Pclass[1]*100/(titanic_df_Surv_Sex_Pclass[0]+titanic_df_Surv_Sex_Pclass[1]),1)
titanic_df_Surv_Sex_Pclass['Surviving_rate'].plot(kind='bar', title='surviving rate per Sex and Pclass',figsize=(16,10))
plt.ylabel('%')
plt.show()


# #### 6 Something else interestiong 
#     determine Pearson Correlation for a variable

# In[510]:


def Strong_pearson_correlation(df=titanic_df,min_r=0.5):
# determine the r_pearson coorelation matrix
    corr_pearson=df.corr(method='pearson')
    # get correlation at min_r
    my_strong_corr={} 
    for col in corr_pearson.columns:
        for row in corr_pearson.index:
            if row==col:
                break
            else:
                if abs(corr_pearson.loc[row, col])>=min_r and abs(corr_pearson.loc[row, col])<1:
                    corr=(row,col)
                    my_strong_corr[corr]=round(corr_pearson.loc[row, col],6)
    return my_strong_corr

def find_pearson_correlation(df=titanic_df, var='Survived',min_r=0.5):
    ''' returns a list of correlation for a given var
    inputs:
        - Var (string) : variable that is investigated
        - min_r (float) : min r pearson for coorelation to be relevant
    returns:
        - list of correlation and associated   r pearson
    default input
        - var = 'EXITSn'
        - min_r=0.6
    examples:
        - find_pearson_correlation() #for default value
        - find_pearson_correlation('EXITSn',0.7)
        - find_pearson_correlation('EXITSn',0.5)
        
        
    ''' 
    if type(var)!=str:
        print('the variable must be between " " or ' ' and be a string')
    else:
        my_strong_corr=Strong_pearson_correlation(df=df,min_r=min_r)
                        
         # get the correlation involving the given var               
        my_corr=[]
        for i in range(len(list(my_strong_corr.keys()))):
            if var in list(my_strong_corr.keys())[i]:
                my_corr.append(list(my_strong_corr.keys())[i])
                
        my_related_corr={}
        for corr in my_corr:
            my_related_corr[corr]=my_strong_corr[corr]
        return my_related_corr


# In[511]:


def gender_to_int(Sex):
    if Sex=='female':
        return 1
    elif Sex=='male':
        return 0

def port_to_int(port):
    if port=='C':
        return 0
    elif port=='Q':
        return 1
    elif port=='S':
        return 2 
    
titanic_df['gender_code']=titanic_df.Sex.apply(gender_to_int)
titanic_df['port_code']=titanic_df.Embarked.apply(port_to_int)
#titanic_df.head()


# In[512]:


print(Strong_pearson_correlation(df=titanic_df,min_r=0.5))
print(find_pearson_correlation(df=titanic_df, var='Survived',min_r=0.5))
print(find_pearson_correlation(df=titanic_df, var='Pclass',min_r=0.5))


# ####  Fare vs Embarked Port and Pclass

# In[513]:


Pclass_vs_fare=titanic_df.groupby(['Ticket','Fare','Pclass','Embarked'],as_index=False)[['Fare','Pclass','Embarked']].all()
#Pclass_vs_fare2=Pclass_vs_fare.groupby(['Pclass','Embarked'],as_index=False)['Fare'].mean()
Pclass_vs_fare2=Pclass_vs_fare.groupby(['Pclass','Embarked'],as_index=False)['Fare'].sum()
first=Pclass_vs_fare2[Pclass_vs_fare2.Pclass==1].set_index('Embarked')
second=Pclass_vs_fare2[Pclass_vs_fare2.Pclass==2].set_index('Embarked')
third=Pclass_vs_fare2[Pclass_vs_fare2.Pclass==3].set_index('Embarked')
ind=third.index
tiket_price_df=pd.concat([first['Fare'], second['Fare'],third['Fare']], axis=1, keys=['first', 'second','third'])
tiket_price_df.plot(kind='bar')
plt.ylabel('Mean Fare')
plt.show()


# In[514]:


titanic_df_by_tiket=titanic_df.groupby(['Ticket','Fare','Pclass','Embarked'],as_index=False)[['Fare','Pclass','Embarked']].size().reset_index(name='Pcount')
tiket_stat_df=titanic_df_by_tiket.groupby(['Embarked','Pclass'],as_index=False)['Fare'].agg({'stat': 'describe'})
tiket_stat_df


# In[515]:


print('\nFare volatility\n')
#titanic_df.boxplot(column='Fare',by=['Pclass','Embarked'],figsize=(16,6))
titanic_df.boxplot(column='Fare',by=['Embarked'],figsize=(16,6))

plt.show()


# In[516]:


titanic_df.groupby(['Pclass','Survived'],as_index=False)['Fare'].agg({'stat':'describe'})


# ### Conclusion
#      preliminary comment: 
#          this study deals with a bit more than the half of the Titanic passenger list. (891 vs 1352)
#          the titanic crew is not involved in the study. allthough it might affect the reflection about survivors 
#          
#          
#      for the given data, the folowing conclusion can be drawn about the survivors:
#          - Women are more likly to survive 
#          - Children are more likly to survive
#          - Rich are more likly to survive
#          
#      more details:
#          - Women and girls(<18 years old) of the 1st and 2nd Class have a surviving rate of about 90%  
#          - Boys(<18 years old) of the 1st and 2nd Class have a surviving rate of about 80% and 60% respectively
#          - Men (>18) of 1st, 2nd and 3rd Class have the lowest survivinge rate of about 35%, 10% and 15% respectively
#          - interesting fact is that the surviving rate of a 1st class male adult(35%) is higher than 3rd class boy(20%)
#          - Interesting fact about 1st class: survivors have a significanctly higher fare average than the deads
#          in the end the most inpacting variable for on the surviving is gender, age(as defined in the study) and class in this order 
#       
#       other facts:
#           - the way the embarcation port atribute fare to a tiket seems chaotic without further details
#           - fare volatility(std) vs port attendence could be correlated (too less port to conclude)
#          
#         

# In[ ]:





# In[ ]:





# In[ ]:




