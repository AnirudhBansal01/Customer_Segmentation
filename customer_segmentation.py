import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans, AgglomerativeClustering

from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)

data = pd.read_csv('/content/drive/MyDrive/marketing_campaign.csv', sep="\t")

data.isnull().sum()

data = data.dropna()

data.isnull().sum()

data.duplicated().sum()

data.info()

data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], dayfirst=True)

data['Age'] = 2015 - data['Year_Birth']

data['Spent'] = data['MntWines'] + data['MntFruits'] + data['MntMeatProducts'] + data['MntFishProducts'] + data['MntSweetProducts'] + data['MntGoldProds']

data['Living_With'] = data['Marital_Status'].replace({'Married':'Partner', 'Together':'Partner', 'Absurd':'Alone', 'Widow':'Alone', 'YOLO':'Alone', 'Divorced':'Alone', 'Single':'Alone'})

data['Children'] = data['Kidhome'] + data['Teenhome']

data['Family_Size'] = data['Living_With'].replace({'Alone': 1, 'Partner':2}) + data['Children']

data['Is_Parent'] = np.where(data.Children > 0, 1, 0)

data['Education'] = data['Education'].replace({'Basic':'Undergraduate', '2n Cycle':'Undergraduate', 'Graduation':'Graduate', 'Master':'Postgraduate', 'PhD':'Postgraduate'})

to_drop = ['Marital_Status', 'Dt_Customer', 'Z_CostContact', 'Z_Revenue', 'Year_Birth', 'ID']
data = data.drop(to_drop, axis=1)

data.head(3)

data.shape

data.info()

data.describe()

sns.lmplot(x='Income', y='Spent', hue='Children', data=data, palette='husl', height=5, aspect=1.5)
plt.title('Income vs Spent with Children Hue')
plt.show()

sns.lmplot(x='Age', y='Spent', hue='Children', data=data, palette='husl', height=5, aspect=1.5)
plt.title('Age vs Spent with Children Hue')
plt.show()

sns.lmplot(x='Age', y='Income', hue='Children', data=data, palette='husl', height=5, aspect=1.5)
plt.title('Age vs Income with Children Hue')
plt.show()

fig1 = px.scatter(data, x='Income', y='Spent', color='Children',
                  trendline='ols', title='Income vs Spent',
                  labels={'Income': 'Income', 'Spent': 'Total Spending'},
                  color_continuous_scale=px.colors.sequential.Viridis)
fig1.show()

plt.figure(figsize=(13,8))
sns.distplot(data.Age, color='purple');

plt.figure(figsize=(13,8))
sns.distplot(data.Income, color='Yellow');

plt.figure(figsize=(13,8))
sns.distplot(data.Spent, color='#ff9966');

fig = make_subplots(rows=1, cols=3)

fig.add_trace(go.Box(y=data['Age'], notched=True, name='Age', marker_color = '#6699ff',
                     boxmean=True, boxpoints='suspectedoutliers'), 1, 2)

fig.add_trace(go.Box(y=data['Income'], notched=True, name='Income', marker_color = '#ff0066',
                     boxmean=True, boxpoints='suspectedoutliers'), 1, 1)

fig.add_trace(go.Box(y=data['Spent'], notched=True, name='Spent', marker_color = 'lightseagreen',
                     boxmean=True, boxpoints='suspectedoutliers'), 1, 3)

fig.update_layout(title_text='Box Plots for Numerical Variables')

fig.show()

numerical = ['Income', 'Recency', 'Age', 'Spent']

def detect_outliers(d):
  for i in d:
    Q3, Q1 = np.percentile(data[i], [75 ,25])
    IQR = Q3 - Q1

    ul = Q3+1.5*IQR
    ll = Q1-1.5*IQR

    outliers = data[i][(data[i] > ul) | (data[i] < ll)]
    print(f'*** {i} outlier points***', '\n', outliers, '\n')

detect_outliers(numerical)

data = data[(data['Age']<100)]
data = data[(data['Income']<600000)]

categorical = [var for var in data.columns if data[var].dtype=='O']

for var in categorical:
    print(data[var].value_counts() / float(len(data)))
    print()
    print()

data['Education'] = data['Education'].map({'Undergraduate':0,'Graduate':1, 'Postgraduate':2})

data['Living_With'] = data['Living_With'].map({'Alone':0,'Partner':1})

data.dtypes

corrmat = data.corr()


plt.figure(figsize=(20, 20))

sns.heatmap(corrmat,
            annot=True,
            cmap='coolwarm',
            center=0,
            linewidths=0.5,
            linecolor='white',
            annot_kws={"size": 10}
            )


plt.title('Correlation Matrix of Customer Personality Analysis Dataset', fontsize=18)
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12)

# Show the heatmap
plt.show()

data_old = data.copy()

cols_del = ['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1','AcceptedCmp2', 'Complain', 'Response']
data = data.drop(cols_del, axis=1)

scaler = StandardScaler()
data = pd.DataFrame(scaler.fit_transform(data), columns = data.columns)

p = PCA(n_components=3)
p.fit(data)

W = p.components_.T
W

pd.DataFrame(W, index=data.columns, columns=['W1','W2','W3'])

pd.DataFrame(p.explained_variance_ratio_, index=range(1,4), columns=['Explained Variability'])

data_PCA = pd.DataFrame(p.transform(data), columns=(['col1', 'col2', 'col3']))

x = data_PCA['col1']
y = data_PCA['col2']
z = data_PCA['col3']

fig = plt.figure(figsize=(13, 8))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(x, y, z, c=z, cmap='viridis', s=50, marker='o', alpha=0.8, edgecolor='k')

cbar = plt.colorbar(sc)
cbar.set_label('Col3 Value (Color Mapping)', fontsize=12)

ax.set_xlabel('Principal Component 1', fontsize=12)
ax.set_ylabel('Principal Component 2', fontsize=12)
ax.set_zlabel('Principal Component 3', fontsize=12)
ax.set_title('Enhanced 3D Projection of Data in Reduced Dimension', fontsize=15)

ax.view_init(elev=20, azim=120)

plt.show()

Elbow_M = KElbowVisualizer(KMeans(), k=10)
Elbow_M.fit(data_PCA)
Elbow_M.show();

AC = AgglomerativeClustering(n_clusters=4)
yhat_AC = AC.fit_predict(data_PCA)
data_PCA['Clusters'] = yhat_AC
data['Clusters'] = yhat_AC
data_old['Clusters'] = yhat_AC

x = data_PCA.iloc[:, 0]
y = data_PCA.iloc[:, 1]
z = data_PCA.iloc[:, 2]

fig = plt.figure(figsize=(13, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(x, y, z, s=40, c=data_PCA['Clusters'], marker='o', cmap='Set1_r')

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Cluster')

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')

ax.set_title('3D Clustering Visualization')

plt.show()

pal = ['gold', '#cc0000', '#ace600', '#33cccc']

plt.figure(figsize=(13, 8))
pl = sns.countplot(x=data['Clusters'], palette=pal)

pl.set_title('Distribution of Clusters', fontsize=16)
pl.set_xlabel('Cluster', fontsize=14)
pl.set_ylabel('Number of Instances', fontsize=14)

for p in pl.patches:
    pl.annotate(format(p.get_height(), '.0f'),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha = 'center', va = 'center',
                xytext = (0, 9),
                textcoords = 'offset points',
                fontsize=12, color='black')


plt.show()

pal = ['gold', '#cc0000', '#ace600', '#33cccc']

plt.figure(figsize=(13, 8))

sns.swarmplot(x=data_old['Clusters'], y=data_old['Spent'], color="#CBEDDD", alpha=0.7, size=8, edgecolor=None)

sns.boxenplot(x=data_old['Clusters'], y=data_old['Spent'], palette=pal, showfliers=False, linewidth=1.5)

plt.title('Distribution of Spending by Cluster', fontsize=16)
plt.xlabel('Cluster', fontsize=14)
plt.ylabel('Amount Spent', fontsize=14)

plt.show()

pal = ['gold', '#cc0000', '#ace600', '#33cccc']

data_old['Total_Promos'] = (
    data_old['AcceptedCmp1'] +
    data_old['AcceptedCmp2'] +
    data_old['AcceptedCmp3'] +
    data_old['AcceptedCmp4'] +
    data_old['AcceptedCmp5']
)

plt.figure(figsize=(13, 8))
pl = sns.countplot(x=data_old['Total_Promos'], hue=data_old['Clusters'], palette=pal)

pl.set_title('Count of Accepted Promotions by Cluster', fontsize=16)
pl.set_xlabel('Number of Total Accepted Promotions', fontsize=14)
pl.set_ylabel('Count', fontsize=14)

plt.legend(title='Cluster', loc='upper right', title_fontsize='13', fontsize='11')

plt.show()

pal = ['gold', '#cc0000', '#ace600', '#33cccc']

plt.figure(figsize=(13, 8))
pl = sns.boxenplot(x=data_old['Clusters'], y=data_old['NumDealsPurchases'], palette=pal)

pl.set_title('Distribution of Deals Purchased by Cluster', fontsize=16)
pl.set_xlabel('Cluster', fontsize=14)
pl.set_ylabel('Number of Deals Purchased', fontsize=14)

pl.grid(True, linestyle='--', alpha=0.7)
pl.tick_params(axis='both', which='major', labelsize=12)

for i, patch in enumerate(pl.patches):
    median = patch.get_data()[1].median()
    pl.text(i, median, f'{int(median)}', ha='center', va='center', color='black', fontsize=12)

plt.show()

Personal = ['Kidhome', 'Teenhome', 'Age', 'Children', 'Family_Size', 'Is_Parent', 'Education', 'Living_With']

for i in Personal:
    plt.figure(figsize=(13,8))
    sns.jointplot(x=data_old[i], y=data_old['Spent'], hue=data_old['Clusters'], kind='kde', palette=pal);