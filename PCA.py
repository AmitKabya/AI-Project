import importlib
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
importlib.import_module('mpl_toolkits.mplot3d').Axes3D


def pca_predict(X, y=None, dim=3):
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=dim)
    pca.fit_transform(X)
    result = pd.DataFrame(data=pca.transform(X), columns=['PCA%i' % i for i in range(dim)])
    if y is not None:
        result.insert(0, 'classes', y)
    return result


def plot_pca(pca, fig):
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    edible = pca['classes'] == 1
    ax.scatter(pca.loc[edible, 'PCA0'], pca.loc[edible, 'PCA1'], pca.loc[edible, 'PCA2'], c='g', cmap="Set2_r", s=10)

    poisonous = pca['classes'] != 1
    ax.scatter(pca.loc[poisonous, 'PCA0'], pca.loc[poisonous, 'PCA1'], pca.loc[poisonous, 'PCA2'], c='r', cmap="Set2_r", s=10)

    ax.legend(['edible', 'poisonous'])
    ax.grid()
