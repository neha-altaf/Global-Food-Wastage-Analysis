import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#load dataset
dataset=pd.read_excel(r'C:\Users\DELL\Desktop\Global food wastage python project\food_waste_dataset.xlsx')
print(dataset)

         #INSPECTING DATASET
print(dataset.info())
print(dataset[['Total Waste (Tons)','Economic Loss (Million $)','Avg Waste per Capita (Kg)']].describe())

print(dataset.isnull().sum())

print(dataset.duplicated().sum())

        #EXPLORATORY DATA ANALAYSIS

#Country wise total food waste analysis(top 10 countries)

top10_countries=dataset.groupby('Country')['Total Waste (Tons)'].sum().sort_values(ascending=False).head(10)

print(top10_countries)

ax=top10_countries.plot(kind='bar',color='grey', figsize=(10,6))
ax.bar_label(ax.containers[0],color='white')
plt.title('Top 10 Countries by Total Food Waste')
plt.gca().set_facecolor('green')
plt.ylabel('Total Waste (Tons)')
plt.yticks([])
plt.xticks(rotation=25)
plt.xlabel("")
plt.show()


#food category wise waste analysis

fc=dataset.groupby('Food Category')['Total Waste (Tons)'].sum().sort_values(ascending=False)

print(fc)

bx=fc.plot(kind='bar',color='yellow', figsize=(10,6))
bx.bar_label(bx.containers[0],color='white')
plt.title('Food Category wise Total Wastage')
plt.gca().set_facecolor('green')
plt.ylabel('Total Waste (Tons)')
plt.yticks([])
plt.xticks(rotation=15)
plt.xlabel("")
plt.show()
#this shows prepared food have largest contribution in total wastage

#Food categories with highest economic loss

FC_with_HEL=dataset.groupby('Food Category')['Economic Loss (Million $)'].sum().sort_values(ascending=False)
print(FC_with_HEL)

FC_with_HEL.plot(kind='line',color='yellow',marker='o', figsize=(12,6),zorder=2)
plt.title('Food Categories with Respective Economic Loss')
plt.grid(axis='both',linestyle='--',color='black', zorder=1)
plt.gca().set_facecolor('green')
plt.ylabel('Economic Loss (Million $)')
plt.xticks(rotation=15)
plt.xlabel("")
plt.show()

#Food categories with respective household waste

FC_with_HHW=dataset.groupby('Food Category')['Household Waste (%)'].median().sort_values(ascending=False)
print(FC_with_HHW)

FC_with_HHW.plot(kind='line',color='yellow',marker='o', figsize=(12,6),zorder=2)
plt.title('Food Categories with Respective HouseHold waste(%)')
plt.grid(axis='both',linestyle='--',color='black', zorder=1)
plt.gca().set_facecolor('green')
plt.ylabel('Household Waste (%)')
plt.xticks(rotation=15)
plt.xlabel("")
plt.show()


#Economic loss analysis

economic_loss=dataset.groupby('Country')['Economic Loss (Million $)'].sum().sort_values(ascending=False).head(10)

print(economic_loss)

cx=economic_loss.plot(kind='bar',color='blue', figsize=(10,6))
cx.bar_label(cx.containers[0],color='white')
plt.title('Top 10 countries in economic loss')
plt.gca().set_facecolor('green')
plt.ylabel('Economic loss (Million $)')
plt.xticks(rotation=15)
plt.yticks([])
plt.xlabel("")
plt.show()

#Average waste per capita by country

AWPC=dataset.groupby('Country')['Avg Waste per Capita (Kg)'].sum().sort_values(ascending=False).head(10)

print(AWPC)

dx=AWPC.plot(kind='bar',color='blue', figsize=(10,6))
dx.bar_label(dx.containers[0],color='white')
plt.title('Top 10 countries in Average waste per capita')
plt.gca().set_facecolor('black')
plt.ylabel('Avg waste per capita(kg)')
plt.xticks(rotation=15)
plt.yticks([])
plt.xlabel("")
plt.show()


#Time analysis

yearwise=dataset.groupby('Year')['Total Waste (Tons)'].sum()
print(yearwise)

yearwise.plot(kind='line', marker ='o',zorder=2,color='white')
plt.gca().set_facecolor('black')
plt.title('Food Wastage Yearly Analysis')
plt.grid(axis='both', linestyle='--', alpha =0.5, color='grey', zorder=1)
plt.xlabel("")
plt.ylabel('Food wastage in Tons')
plt.show()
#this shows 2020 was peak year in food  wastage
#lets dive little bit more deeper which country and food catrgory were on top in 2020
#extract 2020 record
year_2020=dataset[dataset['Year']==2020]
print(year_2020)

top_10C_in_2020=year_2020.groupby('Country')['Total Waste (Tons)'].sum().sort_values(ascending=False).head(10)
print(top_10C_in_2020)
ex=top_10C_in_2020.plot(kind='bar',color='red', figsize=(10,6))
ex.bar_label(ex.containers[0],color='white')
plt.title('Top 10 countries in 2020 by Food wastage')
plt.gca().set_facecolor('black')
plt.ylabel('Total Waste (Tons)')
plt.xticks(rotation=15)
plt.yticks([])
plt.xlabel("")
plt.show()
#this shows india, china and spain were in top 3 in 2020 total food wastage


top_FC_in_2020=year_2020.groupby('Food Category')['Total Waste (Tons)'].sum().sort_values(ascending=False).head(10)
print(top_FC_in_2020)

top_FC_in_2020.plot(kind='line',color='red',marker='o', figsize=(12,6),zorder=2)
plt.title('Top 10 Food Categories in 2020 by Food wastage')
plt.grid(axis='both',linestyle='--',color='grey', zorder=1)
plt.gca().set_facecolor('black')
plt.ylabel('Total Waste (Tons)')
plt.xticks(rotation=15)
plt.xlabel("")
plt.show()
#this tells us that Dairy products were most wasted in 2020


#Relationship between average waste per capita and economic loss
AWPC_vs_ELM=dataset[['Avg Waste per Capita (Kg)','Economic Loss (Million $)']].corr()
print(AWPC_vs_ELM)

sns.heatmap(AWPC_vs_ELM,annot=True)
plt.show()


            #----------------#
            #MACHINE LEARNING
            #----------------#

#SENSITIVITY ANALYSIS
#total wastage vs household wastage analysis
#convert household waste from % to tons

dataset['household waste (tons)']=(dataset['Total Waste (Tons)']*dataset['Household Waste (%)'])/100
 
rel_bw_TWHW=dataset[['Total Waste (Tons)','household waste (tons)']].corr()
print(rel_bw_TWHW)

sns.heatmap(rel_bw_TWHW,annot=True)
plt.show()
#this shows there is strong positive relationship between both variables

#now we will observe how much change in total waste (tons) will occur by change in
#household waste

#first we will check for outliners as we will use linear regression analysis and
#it's better to remove outliners

sns.boxplot(x=dataset['Total Waste (Tons)'])
plt.show()
#remove outliners
Q1=dataset['Total Waste (Tons)'].quantile(0.25)
Q3=dataset['Total Waste (Tons)'].quantile(0.75)
IQR=Q3-Q1
min_range=Q1-(1.5*IQR)
max_range=Q3+(1.5*IQR)

newdataset=dataset.loc[(dataset['Total Waste (Tons)']<=max_range)]
print(newdataset)

#Linear Regression Model

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X=newdataset[['household waste (tons)']]
y=newdataset[['Total Waste (Tons)']]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(X_train,y_train)

print(model.score(X_test,y_test))
print(f"Coefficient of Determination (R-squared):{model.score(X,y)}")
print(f"Slope (coefficient):{model.coef_[0]}")
print(f"Intercept:{model.intercept_}")

slope=model.coef_[0]
intercept=model.intercept_

x_values=np.linspace(newdataset['household waste (tons)'].min() , newdataset['household waste (tons)'].max(),100)
y_values=slope*x_values+intercept

plt.scatter(newdataset['household waste (tons)'],newdataset['Total Waste (Tons)'])
plt.plot(x_values,y_values,color='red')
plt.xlabel('Household Waste (Tons)')
plt.ylabel('Total Waste (Tons)')
plt.gca().set_facecolor('black')
plt.title('Impact of House Hold Waste on Total Waste')
plt.show()
#this analysis shows for every one ton increase/decrease in household waste will
#increase/decrease total waste by 1.8 tons 


#prediction of economic loss based on avg waste per capita and population
P=newdataset[['Avg Waste per Capita (Kg)','Population (Million)']]
q=newdataset[['Economic Loss (Million $)']]

noodel=LinearRegression()
noodel.fit(P,q)

print('Coefficients:',noodel.coef_)
print(f"Coefficient of Determination (R-squared):{noodel.score(P,q)}")
print(f"Intercept:{noodel.intercept_}")
#this shows for every 1 kg increase in avg waste per capita, the economic loss
#is expected to be increase by 0.15 million & for every one million increase
#in population, economic loss is expected to be increase by  0.03 million

#lets visualize this result
coef1=0.155
coef2=0.038
u=np.linspace(newdataset['Avg Waste per Capita (Kg)'].min(),newdataset['Avg Waste per Capita (Kg)'].max())
v=np.linspace(newdataset['Population (Million)'].min(),newdataset['Population (Million)'].max())
U,V=np.meshgrid(u,v)

z=coef1*U +coef2*V
plt.figure(figsize=(12,6))
plt.contourf(U,V,z,cmap='viridis')
plt.colorbar(label='Economic loss (millions)')
plt.xlabel('Avg waste per capita (kg)')
plt.ylabel('Population(million)')
plt.title('Future impact of population & Avg waste per capita on Economic  loss')
plt.show()

#Decision tree regressor:
from sklearn.tree import DecisionTreeRegressor
M=newdataset[['Avg Waste per Capita (Kg)','Population (Million)']]
n=newdataset[['Economic Loss (Million $)']]

M_train,M_test,n_train,n_test=train_test_split(M,n,test_size=0.2,random_state=42)
fodel=DecisionTreeRegressor()
fodel.fit(M_train,n_train)
print('Accuracy:',fodel.score(M_test,n_test))

print('FeatureImportances:',fodel.feature_importances_)
#this gives feature importance for avg waste per capita is 0.21 which means avg
# waste per capita is responsible for almost 21% of the predicted economic loss
#while feature importance for population is 0.78 which shows population is
#responsible for approximately 78% of the predicted economic loss with 100% accuracy
#lets visualize this result

F1=fodel.feature_importances_
F2=F1*100
variable_names=["Avg Waste \n per Capita",'Population']
plt.figure(figsize=(8,4))
plt.barh(range(len(F2)),F2,color='green')
plt.xlabel('Relative Contribution')
plt.yticks(range(len(variable_names)),variable_names)
for i,value in enumerate (F2):
    plt.text(value+1,i,f'{value:.0f}%',rotation=90,color='white')
    plt.xticks([])
plt.gca().set_facecolor('black')
plt.title('Contribution of Each Feature to Economic Loss Prediction')
plt.show()
