import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


def preprocess_data(data):
    df = data
    cache = {}
    for c in data.columns:
        if df[c].dtype == 'O' and c != 'Segmentation':
            print('O:', c)
            df = pd.get_dummies(df, prefix=[c], columns=[c], drop_first=False)
    lbl = LabelEncoder()
    lbl.fit(list(df.Segmentation.values))
    df.Segmentation = lbl.transform(df.Segmentation.values)
    cache['Segmentation'] = lbl
    df.dropna(axis="rows", how="any", inplace=True)
    return df, cache


def tsne_plot(targets, outputs, save_dir=None):
    # targets = targets.reshape(-1, 1)
    # outputs = outputs.reshape(-1, 1)
    # num_class = np.unique(outputs)
    print('generating t-SNE plot...')
    # tsne_output = bh_sne(outputs)
    tsne = TSNE(random_state=0)
    tsne_output = tsne.fit_transform(outputs)

    df = pd.DataFrame(tsne_output, columns=['x', 'y'])
    df['targets'] = targets

    plt.rcParams['figure.figsize'] = 10, 10
    sns.scatterplot(
        x='x', y='y',
        hue='targets',
        palette=sns.color_palette("hls", 4),
        data=df,
        marker='o',
        legend="full",
        alpha=0.5
    )

    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    print('done!')


def plot3d(targets, outputs):
    tsne = TSNE(n_components=3, random_state=0)
    tsne_output = tsne.fit_transform(outputs)

    # projections = tsne.fit_transform(X_test )

    fig = px.scatter_3d(
        tsne_output, x=0, y=1, z=2,
        color=targets, labels={'color': 'targets'}
    )
    fig.update_traces(marker_size=8)
    fig.show()
