import pandas as pd
from sklearn import manifold, cluster, preprocessing, preprocessing
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15,15)

# Load data
data = pd.read_csv('pokemon_dataset.csv')
print('Dataset shape: ', data.shape)

# Fill null values
data.Type_2 = data.Type_2.fillna('Empty')
data.Egg_Group_2 = data.Egg_Group_2.fillna('Empty')
data.Pr_Male = data.Pr_Male.fillna(50) # prob 50-50

# Encode strings to values
lb_make = preprocessing.LabelEncoder()
data["Type_1"] = lb_make.fit_transform(data["Type_1"])
data["Type_2"] = lb_make.fit_transform(data["Type_2"])
data["isLegendary"] = lb_make.fit_transform(data["isLegendary"])
data["Color"] = lb_make.fit_transform(data["Color"])
data["hasGender"] = lb_make.fit_transform(data["hasGender"])
data["Egg_Group_1"] = lb_make.fit_transform(data["Egg_Group_1"])
data["Egg_Group_2"] = lb_make.fit_transform(data["Egg_Group_2"])
data["hasMegaEvolution"] = lb_make.fit_transform(data["hasMegaEvolution"])
data["Body_Style"] = lb_make.fit_transform(data["Body_Style"])

# Standar scaler and normalizer
data_final = data.iloc[:,2:]
scaler = preprocessing.StandardScaler() 
X_scaled = scaler.fit_transform(data_final) 
X_normalized = preprocessing.normalize(X_scaled) 

# TSNE projection - drop to 2 dimensions
print("Computing TSNE embedding")
X_iso = manifold.TSNE(perplexity=10.0, n_components=2, learning_rate=100).fit_transform(X_normalized)
print("Done.")
print("Embedding space:", X_iso.shape)

# Spectral clustering
print("Computing Spectral Clusters")
spectral = cluster.SpectralClustering(n_clusters=4, affinity="rbf", gamma=1/X_iso[:,0].shape[0] * 10)
clusters = spectral.fit(X_iso)
print("Done.")

LABEL_COLOR_MAP = {0 : 'r', 1 : 'b', 2 : 'c', 3 : 'y'}
label_color = [LABEL_COLOR_MAP[l] for l in clusters.labels_]

legendary_cluster = []
with plt.style.context('fivethirtyeight'):      
    plt.title("Spectral Clustering")
    plt.scatter(X_iso[:, 0], X_iso[:, 1], c=label_color, s=50, cmap='viridis')
    for i, txt in enumerate(data['Name']):
        if data['isLegendary'][i] == 1:
            legendary_cluster.append([X_iso[i, 0], X_iso[i, 1]])
        if i % 15 == 0:
            plt.annotate(txt, (X_iso[i, 0], X_iso[i, 1]), color='black')

plt.annotate('Legendary', 
                [sum(x)/len(x) for x in zip(*legendary_cluster)],
                horizontalalignment='center',
                verticalalignment='center',
                size=20, weight='bold',
                color='#630C3A')
plt.show()
