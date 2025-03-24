## Global Food Wastage Analysis Project

## Overview
This project provides an in-depth analysis of global food wastage using Python. The dataset includes key metrics such as total food waste, economic loss, average waste per capita, household waste percentages, and more. Various data analysis and machine learning techniques are applied to derive meaningful insights and trends.

## Features
 Data Cleaning & Preprocessing: Handling missing values, duplicate data, and outliers.

 Exploratory Data Analysis (EDA): Insights into food waste trends by country, food category, and economic impact.

 Visualization: Bar charts, line graphs, and heatmaps to showcase trends.

 Statistical Analysis: Correlation analysis between different metrics.

 Machine Learning Models:
Linear Regression for predicting total waste based on household waste.
Multiple Linear Regression for forecasting economic loss.
Decision Tree Regressor for feature importance analysis.

# Dataset
The dataset contains:
Country-wise and year-wise food wastage data
Economic loss associated with food wastage
Household waste percentages
Population statistics
Food category-based waste breakdown

## Exploratory Data Analysis (EDA)
# Yearly Analysis of Total Food Wastage:
Peak in 2020 (~37,000 tons) due to COVID-19 disruptions.
Sharp decline in 2021 (~30,000 tons), likely from improved food management.
Fluctuations in 2022-2023, with minor increases and decreases.
Rise again in 2024 (~34,000 tons), indicating a rebound in food consumption.

# Top 10 countires in Total Wastage:
China (38,333.8 tons) and India (36,950.2 tons) are on the Lead in total wastage.
Remaining countries are
Spain (19,884 tons), USA (18,960.5 tons), Turkey (17,687.5 tons), Saudi Arabia (17,169.3 tons), Argentina (17,010.9 tons), 
South Korea (16,788.1 tons), UK (16,011.8 tons), and Indonesia (6,749.78 tons)

# Top 10 countries with the highest economic loss (in million $) due to food waste:
China ($19,166.9M) and India ($18,475.1M) face the most significant losses.
Spain ($9,941.98M), USA ($9,480.23M), Turkey ($8,843.75M), Saudi Arabia ($8,584.67M), Argentina ($8,505.44M), 
South Korea ($8,394.07M), UK ($8,005.88M), and Indonesia ($3,374.89M) follow.

# Top 10 countries in Average waste per capita:
Germany (30,597.5 kg) leads with the highest per capita food waste.
China (27,148.6 kg), Spain (26,689.3 kg), USA (26,663.6 kg), Italy (26,411.5 kg), India (26,205.8 kg),
Brazil (26,090 kg), Mexico (26,036.1 kg), Canada (25,789.9 kg), and South Korea (25,500.7 kg) follow closely.

# Top 10 countries with the highest food wastage in 2020:
India (7,355.94 tons) and China (7,121.48 tons) had the highest food wastage.
Spain (3,736.43 tons), Turkey (3,219.72 tons), Argentina (2,458.69 tons), South Korea (2,417.05 tons),
Saudi Arabia (2,303.81 tons), UK (2,069.32 tons), USA (2,035.71 tons), and Indonesia (1,138.49 tons) follow.

# Top 10 food categories with the highest wastage in 2020:
Dairy Products & Beverages had the highest food wastage, exceeding 5000 tons.
Meat & Seafood, Fruits & Vegetables, and Prepared Food also faced significant wastage.
Bakery Items and Grains & Cereals had moderate wastage levels.
Frozen Food had the least wastage among the listed categories, but still significant.

# Food Categories with Respective HouseHold waste(%):
Bakery Items have the highest household wastage at around 51.7%.
Dairy Products and Fruits & Vegetables also experience high wastage, likely due to spoilage and short shelf life.
Prepared Food and Frozen Food have moderate wastage levels.
Meat & Seafood, Beverages, and Grains & Cereals show lower household waste percentages.

# Food categories with their respective economic loss:
Prepared Food and Frozen Food result in the highest economic losses, exceeding $16,000 million and $15,500 million, respectively.
Fruits & Vegetables, Bakery Items, and Dairy Products also show significant losses, ranging from $13,500 million to $14,500 million.
Grains & Cereals have the lowest economic loss, yet they still contribute notably to overall waste.

# Food Category-wise Total Wastage:
Prepared Food has the highest wastage at 32,583.4 tons, followed closely by Frozen Food (31,250.1 tons).
Fruits & Vegetables, Bakery Items, and Dairy Products also contribute significantly, each exceeding 27,000 tons of waste.
Grains & Cereals have the lowest wastage among the categories but still account for 26,728.6 tons of discarded food.

## MACHINE LEARNING INSIGHTS

# Linear Regression Analysis:
The Model achieved an accuracy of 91%.  
A strong positive correlation (R² = 0.91) between total waste and household waste.

Every 1-ton increase in household waste results in a 1.8-ton increase in total waste.

# Economic Loss Prediction:
Model achieved the accuracy of 81%.

A 1 kg increase in average waste per capita increases economic loss by $0.15 million.

A 1 million population increase raises economic loss by $0.03 million.

# Decision Tree Regressor:
The Model achieved 99% accuracy.
Population contributes 79% to economic loss prediction, while average waste per capita contributes 21%.

## Results & Insights:
Prepared foods and dairy products are the most wasted food categories.

Economic loss is directly linked to population size and per capita food wastage.

Countries with larger populations tend to have higher food wastage and economic losses.

Machine learning models can effectively predict economic loss based on food waste data.

## Recommendations
Reduce Household Waste: Encourage households to buy only necessary perishable items and improve food storage practices.

Improve Supply Chain Management: Efficient food distribution strategies can minimize food spoilage before it reaches consumers.

Food Redistribution Programs: Establish programs to donate surplus food to those in need instead of discarding it.

Awareness Campaigns: Educate consumers about the economic and environmental impact of food wastage.

Policy Interventions: Governments should introduce policies to monitor and control food waste at both industrial and household levels.

## License
This project is for personal and educational purposes only. Unauthorized editing, redistribution, or commercial use is prohibited.

## Requirements:
1. Software Requirements
Python 3.x (Recommended: Python 3.7 or later)

Jupyter Notebook (Optional, for interactive execution)

2. Required Python Libraries
Ensure you have the following Python libraries installed before running the code:
pandas numpy matplotlib seaborn scikit-learn openpyxl

3. Dataset Requirement
The project uses an Excel dataset named food_waste_dataset.xlsx. Dataset has been added to repository.
Download the dataset and modify the dataset path in pd.read_excel() to match your local directory structure.

For any queries or discussions, you can reach out via:    
- **LinkedIn**: www.linkedin.com/in/neha-altaf-44952726a  
- **Email**: nehaaltaf24@gmail.com
