import argparse
import os
import sys
import torch
import pickle
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.model import MultimodalEnhancedHyperModel
from tsne_visualization import ModelVisualizer

def visualize_trained_model(model_path, data_path, output_dir='./visualizations', name=''):

    print(f"Loading dataset from {data_path}...")

    dataset = pickle.load(open(data_path, 'rb'))

    print(f"Loading model from {model_path}...")

    model = MultimodalEnhancedHyperModel(
        input_dim=100,  
        output_dim=100,
        max_edge_types=12, 
        text_dim=256
    )

    state_dict = torch.load(model_path)
    print("Saved keys:", state_dict.keys())

    print("Model keys:", model.state_dict().keys())
    model.load_state_dict(torch.load(model_path))
    model.cuda()

    visualizer = ModelVisualizer(output_dir=output_dir)

    print("Extracting features...")
    features, labels, predictions, ids = visualizer.extract_features(
        model, dataset, mode='train'
    )

    print("Performing t-SNE visualization...")
    visualizer.tsne_visualization(
        features, labels, predictions,name,title="Trained Model - Test Set Visualization"
    )

    print(f"Visualization saved to: {output_dir}")


if __name__ == "__main__":
    output_dir = '/data/AIinspur02/linshi001/output/hyper_w2v_7edges_RealHyper0407/tsne_data1'
    name = "multi_model71"

    model_path = '/data/AIinspur02/linshi001/output/hyper_w2v_RealHyper_model3/MultimodalModel40-model.bin'
    data_path = '/data/AIinspur02/linshi001/dataset/hyper_w2v_7edges_RealHyper0403/devign_all_code_codet5p_256.bin'

    visualize_trained_model(model_path, data_path, output_dir, name)