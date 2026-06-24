# Exploratory Data Analysis on Fleet Health Dataset

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load CSV file
df = pd.read_csv("fleet_health.csv")

# Set background style for all charts
sns.set_theme(style='whitegrid')

# ── BASIC DATA UNDERSTANDING ───────────────────────────────────────────────

print("Shape of Dataset (rows, columns):")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nMean of all numeric columns:")
print(df.mean(numeric_only=True))

# ── CHART 1: STATUS DISTRIBUTION ──────────────────────────────────────────

plt.figure(figsize=(8, 5))

sns.countplot(data=df,
              x='Status',
              hue='Status',
              palette={'Truck Healthy': 'green',
                       'Service Required Soon': 'orange',
                       'High Risk': 'red'},
              order=['Truck Healthy', 'Service Required Soon', 'High Risk'],
              legend=False)

plt.title('Fleet Status Distribution', fontsize=16)
plt.xlabel('Status', fontsize=12)
plt.ylabel('Number of Trucks', fontsize=12)

plt.tight_layout()
plt.savefig('Chart1_Status_Distribution.png')
plt.show()
print("Chart 1 saved!")

# ── CHART 2: AGE DISTRIBUTION ─────────────────────────────────────────────

plt.figure(figsize=(8, 5))

sns.histplot(data=df,
             x='Age',
             bins=15,
             color='steelblue',
             kde=True)

plt.title("Truck Age Distribution", fontsize=16)
plt.xlabel('Age in Years', fontsize=12)
plt.ylabel('Number of Trucks', fontsize=12)

plt.tight_layout()
plt.savefig('Chart2_Age_Distribution.png')
plt.show()
print("Chart 2 saved!")

# ── CHART 3: ENGINE TEMP BY STATUS ────────────────────────────────────────

plt.figure(figsize=(8, 5))

sns.boxplot(data=df,
            x='Status',
            y='Engine_Temp',           # fixed: capital T
            hue='Status',
            palette={'Truck Healthy': 'green',
                     'Service Required Soon': 'orange',
                     'High Risk': 'red'},
            order=['Truck Healthy', 'Service Required Soon', 'High Risk'],
            legend=False)

plt.title("Engine Temperature by Fleet Status", fontsize=16)
plt.xlabel("Status", fontsize=12)
plt.ylabel("Engine Temperature in Celsius", fontsize=12)

plt.tight_layout()
plt.savefig('Chart3_Engine_Temp_By_Status.png')
plt.show()
print("Chart 3 saved!")

# ── CHART 4: CORRELATION HEATMAP ──────────────────────────────────────────

plt.figure(figsize=(10, 7))

numeric_df = df.select_dtypes(include='number')
correlation = numeric_df.corr()

sns.heatmap(correlation,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            linewidths=0.5)

plt.title("Correlation Between Features", fontsize=16)

plt.tight_layout()
plt.savefig("Chart4_Correlation_Heatmap.png")
plt.show()
print("Chart 4 saved!")

# ── CHART 5: AGE VS TOTAL KM DRIVEN ───────────────────────────────────────

plt.figure(figsize=(9, 6))

sns.scatterplot(data=df,
                x="Age",
                y="Total_KM_Driven",
                hue='Status',
                palette={'Truck Healthy': 'green',
                         'Service Required Soon': 'orange',
                         'High Risk': 'red'},
                alpha=0.6)             # fixed: alpha not aplha

plt.title("Age vs Total KM Driven (colored by Status)", fontsize=16)
plt.xlabel('Age in Years', fontsize=12)
plt.ylabel('Total KM Driven', fontsize=12)

plt.tight_layout()
plt.savefig("Chart5_Age_vs_KM.png")   # fixed: .png not .ong
plt.show()
print("Chart 5 saved!")