import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px


def scale_features(features):
    """
    Scales the features using Z-score normalization
    :param features:
    """
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    return features_scaled, scaler


def reduce_features(features, n_components):
    """
    Reduces the dimensionality of the features using PCA or LDA
    :param features:
    :param n_components:
    """
    dr_model = PCA(n_components=n_components, random_state=42)
    features_reduced = dr_model.fit_transform(features)
    return features_reduced, dr_model


def visualize_dr(features_reduced, labels, n_components, method):
    """
    Visualize the features, either 3D or 2D
    :param labels:
    :param features_reduced:
    :param n_components:
    :param method:
    """
    labels = np.array(labels)
    unique_labels = np.unique(labels)

    if n_components == 2:
        plt.figure(figsize=(6, 5))
        for lbl in unique_labels:
            idx = np.where(labels == lbl)
            plt.scatter(
                features_reduced[idx, 0],
                features_reduced[idx, 1],
                label=str(lbl),
                alpha=0.7,
                s=4
            )
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title(f'PCA training data for {method} (2D)')
        plt.legend()
        plt.show()

    elif n_components == 3:
        plot_data = {
            'Component 1': features_reduced[:, 0],
            'Component 2': features_reduced[:, 1],
            'Component 3': features_reduced[:, 2],
            'Label': labels
        }

        fig = px.scatter_3d(
            plot_data,
            x='Component 1',
            y='Component 2',
            z='Component 3',
            color='Label',
            title=f'PCA training data for {method} (3D)',
            opacity=0.8,
            height=600
        )

        fig.update_traces(
            marker=dict(
                size=2
            )
        )

        fig.update_layout(
            scene=dict(
                xaxis_title='PCA Component 1',
                yaxis_title='PCA Component 2',
                zaxis_title='PCA Component 3',
            )
        )

        fig.show()
