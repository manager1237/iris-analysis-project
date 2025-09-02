

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import os

# Step 0: Setup
sns.set_style("whitegrid")

# Create a folder to save plots
if not os.path.exists("plots"):
    os.makedirs("plots")

# Step 1: Load Dataset
try:
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    print("Dataset loaded successfully!\n")
except Exception as e:
    print("Error loading dataset:", e)
    exit()

# Step 2: Explore Dataset
print("First 5 rows of the dataset:")
print(df.head(), "\n")

print("Dataset info:")
print(df.info(), "\n")

print("Missing values per column:")
print(df.isnull().sum(), "\n")

print("Summary statistics:")
print(df.describe(), "\n")

# Step 3: Basic Data Analysis
species_mean = df.groupby('species').mean()
print("Mean values by species:")
print(species_mean, "\n")

print("Observation: Setosa generally has smaller petal dimensions than Versicolor and Virginica.\n")

# Step 4: Data Visualization

# 1. Line chart - trend of sepal length across index
plt.figure(figsize=(8,5))
plt.plot(df.index, df['sepal length (cm)'], color='blue', label='Sepal Length')
plt.title('Sepal Length Trend')
plt.xlabel('Index')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.tight_layout()
plt.savefig("plots/sepal_length_trend.png")
plt.close()

# 2. Bar chart - average petal length per species
plt.figure(figsize=(8,5))
sns.barplot(x=species_mean.index, y=species_mean['petal length (cm)'])
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.tight_layout()
plt.savefig("plots/average_petal_length_per_species.png")
plt.close()

# 3. Histogram - distribution of sepal width
plt.figure(figsize=(8,5))
plt.hist(df['sepal width (cm)'], bins=10, color='green', edgecolor='black')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig("plots/sepal_width_distribution.png")
plt.close()

# 4. Scatter plot - sepal length vs petal length
plt.figure(figsize=(8,5))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df, s=100)
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.tight_layout()
plt.savefig("plots/sepal_vs_petal_scatter.png")
plt.close()

print("All plots saved in the 'plots' folder.\n")

# Step 5: Summary Observations
print("Summary of Observations:")
print("- Setosa has smaller petal and sepal sizes.")
print("- Virginica has the largest petal and sepal sizes.")
print("- Scatter plots show clear separation among species.")
print("- The dataset is suitable for classification tasks.\n")
