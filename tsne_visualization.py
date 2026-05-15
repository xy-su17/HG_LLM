import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
import os
import json
from datetime import datetime
import pandas as pd
from matplotlib import cm
from matplotlib.patches import Patch

class ModelVisualizer:
    def __init__(self, output_dir='./visualizations'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        plt.style.use('seaborn')

    def extract_features(self, model, dataset, mode='test'):
        model.eval()

        if mode == 'train':
            num_batches = dataset.initialize_train_batch()
            get_batch = dataset.get_next_train_batch
        elif mode == 'valid':
            num_batches = dataset.initialize_valid_batch()
            get_batch = dataset.get_next_valid_batch
        else: 
            num_batches = dataset.initialize_test_batch()
            get_batch = dataset.get_next_test_batch

        all_features = []
        all_labels = []
        all_predictions = []
        all_ids = []

        with torch.no_grad():
            for _ in range(num_batches):
                graph, targets, ids = get_batch()
                targets = targets.cuda()

                predictions, features = model(graph, cuda=True, mode='eval', return_features=True)

                all_features.append(features.cpu().numpy())
                all_labels.append(targets.cpu().numpy())
                all_predictions.append(predictions.argmax(dim=1).cpu().numpy())
                all_ids.extend(ids)

        features_array = np.vstack(all_features)
        labels_array = np.hstack(all_labels)
        predictions_array = np.hstack(all_predictions)

        return features_array, labels_array, predictions_array, all_ids

    def tsne_visualization(self, features, labels, predictions=None,
                           name='',title='t-SNE Visualization',
                           perplexity=30,
                           n_iter=1000,
                           random_state=42,
                           figsize=(10, 8),
                           point_size=15,
                           alpha=0.4,
                           save_path=None):
        print(f"Starting t-SNE with {len(features)} samples, dimension: {features.shape[1]}")

        if features.shape[1] > 50:
            print("Applying PCA for initial dimensionality reduction...")
            pca = PCA(n_components=min(50, features.shape[1]), random_state=random_state)
            features_reduced = pca.fit_transform(features)
            print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
        else:
            features_reduced = features

        print("Applying t-SNE...")
        tsne = TSNE(n_components=2,
                    perplexity=perplexity,
                    n_iter=n_iter,
                    random_state=random_state,
                    verbose=1)

        tsne_results = tsne.fit_transform(features_reduced)

        unique_labels = np.unique(labels)
        centroid_distance = None

        if len(unique_labels) == 2:
            centers = []
            for label_val in unique_labels:
                mask = (labels == label_val)
                center = tsne_results[mask].mean(axis=0)
                centers.append(center)

            if len(centers) == 2:
                centroid_distance = np.linalg.norm(centers[0] - centers[1])

        overall_center = tsne_results.mean(axis=0)
        distances_to_center = np.linalg.norm(tsne_results - overall_center, axis=1)
        mean_dist = distances_to_center.mean()
        if mean_dist > 0:
            centroid_distance_norm = centroid_distance / mean_dist
        else:
            centroid_distance_norm = centroid_distance
        plt.figure(figsize=figsize)

        colors = ['#40ad3f', '#d62728', '#1f77b4', '#ff7f0e', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        for i, label_val in enumerate(unique_labels):
            mask = (labels == label_val)
            color = colors[i % len(colors)]

            label_val_int = int(round(label_val))
            if label_val_int == 0:
                label_name = 'neutral'
            elif label_val_int == 1:
                label_name = 'vulnerable'

            plt.scatter(tsne_results[mask, 0], tsne_results[mask, 1],
                        c=color, label=label_name, s=point_size, alpha=alpha, edgecolors='none')

        accuracy = np.mean(labels == predictions) * 100

        plt.legend(markerscale=2, fontsize=12)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()

        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f'{name}_cd{centroid_distance:.4f}_cdn{centroid_distance_norm:.4f}_Acc{accuracy:.2f}_{timestamp}.png')

        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
        plt.close()

        print(f"t-SNE visualization saved to: {save_path}")
        if centroid_distance_norm is not None:
            print(f"Centroid Distance: {centroid_distance_norm:.4f}")

        return tsne_results

