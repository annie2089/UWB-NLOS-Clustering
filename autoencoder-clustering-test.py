import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score, completeness_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import umap
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
import logging
from datetime import datetime
import matplotlib as mpl
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
matplotlib.use('Agg')

# Suppress scikit-learn deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.utils.deprecation")

# Set up GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU available, using CPU")


# Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, layer_sizes, latent_dim):
        super(Autoencoder, self).__init__()
        encoder_layers = []
        prev_size = input_dim
        for size in layer_sizes:
            encoder_layers.extend([nn.Linear(prev_size, size), nn.ReLU()])
            prev_size = size
        encoder_layers.append(nn.Linear(prev_size, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        prev_size = latent_dim
        for size in reversed(layer_sizes):
            decoder_layers.extend([nn.Linear(prev_size, size), nn.ReLU()])
            prev_size = size
        decoder_layers.append(nn.Linear(prev_size, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent


# Setup Logging
def setup_logging(output_dir: str) -> None:
    log_file = os.path.join(output_dir, "autoencoder_clustering_log.txt")
    try:
        open(log_file, 'w').close()
        print(f"Created empty log file at: {log_file}")
        handler = logging.FileHandler(log_file, mode='a')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.handlers = []
        logger.addHandler(handler)
        logging.info("Logging initialized for end-to-end clustering with ARI objective")
        handler.flush()
        handler.close()
        print(f"Logging to: {log_file}")
    except Exception as e:
        print(f"Error setting up logging to {log_file}: {e}")
        raise


# Load and Preprocess Data
def load_and_combine_datasets(dataset_files: list, features: list) -> pd.DataFrame:
    dataframes = []
    for file in dataset_files:
        if not os.path.exists(file):
            logging.error(f"Dataset file not found: {file}")
            print(f"Error: Dataset file not found: {file}")
            raise FileNotFoundError(f"Dataset file not found: {file}")
        logging.info(f"Loading file: {file}")
        df = pd.read_csv(file, header=0, sep=",", usecols=features)
        df = df.dropna()
        df['NLOS'] = df['NLOS'].astype(int)
        dataframes.append(df)
    combined_df = pd.concat(dataframes, ignore_index=True)
    logging.info(f"Combined dataset shape: {combined_df.shape}")
    return combined_df


def load_single_dataset(file: str, features: list) -> pd.DataFrame:
    if not os.path.exists(file):
        logging.error(f"Dataset file not found: {file}")
        print(f"Error: Dataset file not found: {file}")
        raise FileNotFoundError(f"Dataset file not found: {file}")
    logging.info(f"Loading file: {file}")
    try:
        df = pd.read_csv(file, header=0, sep=",", usecols=features)
        df = df.dropna()
        df['NLOS'] = df['NLOS'].astype(int)
        if df.empty:
            logging.error(f"Dataset {file} is empty after preprocessing")
            print(f"Error: Dataset {file} is empty after preprocessing")
            raise ValueError(f"Dataset {file} is empty after preprocessing")
        logging.info(f"Dataset {file} shape: {df.shape}")
        return df
    except KeyError as e:
        logging.error(f"Missing features in {file}: {e}")
        print(f"Error: Missing features in {file}: {e}")
        raise KeyError(f"Missing features in {file}: {e}")
    except pd.errors.ParserError:
        logging.error(f"Error parsing CSV file: {file}")
        print(f"Error: Error parsing CSV file: {file}")
        raise ValueError(f"Error parsing CSV file: {file}")


# Normalize Features and Save Scalers
def normalize_features(df: pd.DataFrame, features: list, scaler: StandardScaler = None,
                       scaler_path: str = None) -> tuple:
    raw_features = df[features].values
    if scaler is None:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(raw_features)
        if scaler_path:
            try:
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
                logging.info(f"Saved scaler to {scaler_path}")
            except Exception as e:
                logging.error(f"Failed to save scaler to {scaler_path}: {e}")
                print(f"Error: Failed to save scaler to {scaler_path}: {e}")
                raise
    else:
        scaled_features = scaler.transform(raw_features)
    logging.info(f"Features scaled for {df.shape[0]} samples")
    return scaled_features, scaler


# Get Latent Features in Batches
def get_latent_features(model, data, batch_size=256):
    model.eval()
    latent_features = []
    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
    dataset = torch.utils.data.TensorDataset(data_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                             pin_memory=torch.cuda.is_available())

    with torch.no_grad():
        for batch in dataloader:
            batch_data = batch[0].to(device)
            _, latent = model(batch_data)
            latent_features.append(latent.cpu().numpy())

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(latent_features, axis=0)


# Combine Latent and Selected Features
def combine_features(latent_features, selected_features):
    scaler_latent = StandardScaler()
    scaler_selected = StandardScaler()
    latent_scaled = scaler_latent.fit_transform(latent_features)
    selected_scaled = scaler_selected.fit_transform(selected_features)
    combined = np.hstack((latent_scaled, selected_scaled))
    return combined


# Analyze Clusters
def analyze_clusters(clusters: np.ndarray, true_labels: pd.Series, features: np.ndarray,
                     algorithm: str, output_dir: str, params: dict = None, file_name: str = "") -> dict:
    stats = {}
    valid_clusters = clusters
    valid_features = features
    valid_labels = true_labels

    stats['n_clusters'] = len(set(valid_clusters) - {-1})
    stats['cluster_sizes'] = pd.Series(clusters).value_counts().to_dict()
    stats['noise_points'] = stats['cluster_sizes'].get(-1, 0)

    df_temp = pd.DataFrame({'Cluster': valid_clusters, 'Label': valid_labels})
    cluster_mapping = {}
    for cluster in set(valid_clusters):
        if cluster != -1:
            cluster_data = df_temp[df_temp['Cluster'] == cluster]['Label']
            if not cluster_data.empty:
                majority_label = cluster_data.mode().iloc[0]
                cluster_mapping[cluster] = majority_label

    cluster_proportions = {}
    for cluster in set(valid_clusters):
        cluster_data = df_temp[df_temp['Cluster'] == cluster]['Label']
        if not cluster_data.empty:
            proportions = cluster_data.value_counts(normalize=True).to_dict()
            cluster_proportions[cluster] = {f"Label={k}": v for k, v in proportions.items()}
            logging.info(f"[{file_name}] Cluster {cluster} proportions: {cluster_proportions[cluster]}")

    stats['cluster_mapping'] = cluster_mapping
    stats['cluster_proportions'] = cluster_proportions

    mapped_labels = np.array([cluster_mapping.get(c, 0) for c in valid_clusters])
    metrics = {
        'ARI': adjusted_rand_score(valid_labels, mapped_labels),
        'Silhouette': silhouette_score(valid_features, valid_clusters) if stats['n_clusters'] > 1 else -1,
        'Completeness': completeness_score(valid_labels, mapped_labels),
        'Accuracy': accuracy_score(valid_labels, mapped_labels),
        'Precision': precision_score(valid_labels, mapped_labels, average='macro', zero_division=0),
        'Recall': recall_score(valid_labels, mapped_labels, average='macro', zero_division=0),
        'F1': f1_score(valid_labels, mapped_labels, average='macro', zero_division=0)
    }

    stats['metrics'] = metrics
    stats['params'] = params
    return stats

def remap_clusters_to_labels(true_labels, cluster_labels):
    """
    Remap cluster labels to best match the true labels using the Hungarian algorithm.
    """
    true_labels = np.array(true_labels)
    cluster_labels = np.array(cluster_labels)

    # Create a confusion matrix
    conf_mat = confusion_matrix(true_labels, cluster_labels)

    # Apply Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-conf_mat)  # maximize matching

    # Build mapping from cluster to true label
    mapping = {col: row for row, col in zip(row_ind, col_ind)}

    # Remap cluster labels
    new_cluster_labels = np.array([mapping.get(label, label) for label in cluster_labels])

    return new_cluster_labels


# Visualize with UMAP
def visualize_umap(features, clusters, true_labels, algorithm: str, output_dir: str, file_name: str):
    plt.rc('font', size=16, family='Times New Roman')  # Set global font size and family
    # Set Times New Roman globally with font size 16
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.size'] = 16
    mpl.rcParams['axes.titlesize'] = 16
    mpl.rcParams['axes.labelsize'] = 16
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['ytick.labelsize'] = 16
    mpl.rcParams['legend.fontsize'] = 16

    try:
        reducer = umap.UMAP(n_components=2, random_state=42, n_jobs=1)
        embedding = reducer.fit_transform(features)

        color_palette = {0: 'blue', 1: 'red', -1: 'gray', 2: 'green', 3: 'orange', 4: 'purple'}
        cluster_colors = [color_palette.get(c, 'black') for c in clusters]
        true_label_colors = [color_palette.get(l, 'black') for l in true_labels]

        plt.figure(figsize=(6, 5))
        scatter = sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=clusters, palette=color_palette,
                                  alpha=0.6, s=50, legend='full')
        # plt.title(f"{algorithm} Clustering - {file_name}")
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        cluster_sizes = pd.Series(clusters).value_counts()
        legend_labels = [f"Cluster {k} (n={cluster_sizes.get(k, 0)})" for k in set(clusters)]
        plt.legend(handles=scatter.legend_.legend_handles, labels=legend_labels,prop={'family': 'Times New Roman', 'size': 16})
        cluster_plot_path = os.path.join(output_dir, f"{file_name}_{algorithm.lower()}_clusters_umap.pdf")
        plt.savefig(cluster_plot_path, bbox_inches='tight')
        plt.close()
        logging.info(f"[{file_name}] Cluster UMAP plot saved to {cluster_plot_path}")

        plt.figure(figsize=(6, 5))
        scatter = sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=true_labels, palette=color_palette,
                                  alpha=0.6, s=50, legend='full')
        # plt.title(f"True Labels - {file_name}")
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        label_sizes = pd.Series(true_labels).value_counts()
        legend_labels = [f"Label {k} (n={label_sizes.get(k, 0)})" for k in set(true_labels)]
        plt.legend(handles=scatter.legend_.legend_handles, labels=legend_labels,prop={'family': 'Times New Roman', 'size': 16})
        true_label_plot_path = os.path.join(output_dir, f"{file_name}_true_labels_umap.pdf")
        plt.savefig(true_label_plot_path, bbox_inches='tight')
        plt.close()
        logging.info(f"[{file_name}] True labels UMAP plot saved to {true_label_plot_path}")
    except Exception as e:
        logging.error(f"Failed to generate UMAP plots for {file_name}: {e}")
        print(f"Warning: Failed to generate UMAP plots for {file_name}: {e}")

# Test on Individual Files
def test_on_individual_files(dataset_files, features, cir_features, selected_features, output_dir, best_algorithm,
                             best_clustering_params, best_params, model_path, cir_scaler_path, selected_scaler_path):
     # Check if files exist
    if not os.path.exists(model_path):
        logging.error(f"Model file not found: {model_path}")
        print(f"Error: Model file not found: {model_path}")
        return
    if not os.path.exists(cir_scaler_path):
        logging.error(f"CIR scaler file not found: {cir_scaler_path}")
        print(f"Error: CIR scaler file not found: {cir_scaler_path}")
        return
    if not os.path.exists(selected_scaler_path):
        logging.error(f"Selected scaler file not found: {selected_scaler_path}")
        print(f"Error: Selected scaler file not found: {selected_scaler_path}")
        return

    # Load model and scalers
    try:
        with open(cir_scaler_path, 'rb') as f:
            cir_scaler = pickle.load(f)
        with open(selected_scaler_path, 'rb') as f:
            selected_scaler = pickle.load(f)
    except Exception as e:
        logging.error(f"Failed to load scalers: {e}")
        print(f"Error: Failed to load scalers: {e}")
        return

    try:
        layer_sizes = [best_params['layer_sizes'][i] for i in range(best_params['n_layers'])]
        model = Autoencoder(input_dim=len(cir_features), layer_sizes=layer_sizes, latent_dim=best_params['latent_dim'])
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        model.eval()
    except Exception as e:
        logging.error(f"Failed to load model from {model_path}: {e}")
        print(f"Error: Failed to load model: {e}")
        return

    try:
        combined_df = load_and_combine_datasets(dataset_files, features)
        combined_scaled_cir_features, _ = normalize_features(combined_df, cir_features, cir_scaler)
        combined_scaled_selected_features, _ = normalize_features(combined_df, selected_features, selected_scaler)
    except Exception as e:
        logging.error(f"Failed to load or normalize combined dataset: {e}")
        print(f"Error: Failed to load or normalize combined dataset: {e}")
        return

    summary_rows = []
    for file in dataset_files:
        file_name = os.path.basename(file).replace('.csv', '')
        print(f"\nTesting on {file_name}...")
        logging.info(f"Testing on {file_name}")

        try:
            df = load_single_dataset(file, features)
            scaled_cir_features, _ = normalize_features(df, cir_features, cir_scaler)
            scaled_selected_features, _ = normalize_features(df, selected_features, selected_scaler)
            latent_features = get_latent_features(model, scaled_cir_features)
            combined_features = combine_features(latent_features, scaled_selected_features)
        except Exception as e:
            logging.error(f"[{file_name}] Failed to preprocess data: {e}")
            print(f"[{file_name}] Error: Failed to preprocess data: {e}")
            continue

        try:
            if best_algorithm == 'KMeans':
                clusterer = KMeans(n_clusters=best_clustering_params['n_clusters'], random_state=42)
            elif best_algorithm == 'GMM':
                clusterer = GaussianMixture(n_components=best_clustering_params['n_components'], random_state=42)
            else:
                clusterer = DBSCAN(eps=best_clustering_params['eps'], min_samples=best_clustering_params['min_samples'])
            clusters = clusterer.fit_predict(combined_features)
            stats = analyze_clusters(clusters, df['NLOS'], combined_features, best_algorithm, output_dir,
                                     best_clustering_params, file_name)
            remapped_clusters = remap_clusters_to_labels(df['NLOS'], clusters)
            visualize_umap(combined_features, remapped_clusters, df['NLOS'], best_algorithm, output_dir, file_name)
            # visualize_umap(combined_features, clusters, df['NLOS'], best_algorithm, output_dir, file_name)
        except Exception as e:
            logging.error(f"[{file_name}] Failed to cluster or visualize: {e}")
            print(f"[{file_name}] Error: Failed to cluster or visualize: {e}")
            continue

        summary_rows.append({
            'File': file_name,
            'Algorithm': best_algorithm,
            'ARI': stats['metrics']['ARI'],
            'Silhouette': stats['metrics']['Silhouette'],
            'Completeness': stats['metrics']['Completeness'],
            'F1': stats['metrics']['F1'],
            'Accuracy': stats['metrics']['Accuracy'],
            'Precision': stats['metrics']['Precision'],
            'Recall': stats['metrics']['Recall'],
            'Clusters': stats['n_clusters'],
            'Noise Points': stats['noise_points']
        })

        metrics_str = ", ".join([f"{k}={v:.3f}" for k, v in stats['metrics'].items()])
        print(f"[{file_name}] Algorithm: {best_algorithm}, Metrics: {metrics_str}")
        print(f"[{file_name}] Clusters: {stats['n_clusters']}, Noise Points={stats['noise_points']}")
        print(f"[{file_name}] Cluster Sizes: {stats['cluster_sizes']}")
        print(f"[{file_name}] Cluster Proportions: {stats['cluster_proportions']}")
        logging.info(f"[{file_name}] Algorithm: {best_algorithm}, Metrics: {metrics_str}")
        logging.info(f"[{file_name}] Clusters: {stats['n_clusters']}, Noise Points={stats['noise_points']}")
        logging.info(f"[{file_name}] Cluster Sizes: {stats['cluster_sizes']}")
        logging.info(f"[{file_name}] Cluster Proportions: {stats['cluster_proportions']}")

    # Test on combined dataset
    try:
        latent_features = get_latent_features(model, combined_scaled_cir_features)
        combined_features = combine_features(latent_features, combined_scaled_selected_features)
        if best_algorithm == 'KMeans':
            clusterer = KMeans(n_clusters=best_clustering_params['n_clusters'], random_state=42)
        elif best_algorithm == 'GMM':
            clusterer = GaussianMixture(n_components=best_clustering_params['n_components'], random_state=42)
        else:
            clusterer = DBSCAN(eps=best_clustering_params['eps'], min_samples=best_clustering_params['min_samples'])
        combined_clusters = clusterer.fit_predict(combined_features)
        stats = analyze_clusters(combined_clusters, combined_df['NLOS'], combined_features, best_algorithm, output_dir,
                                 best_clustering_params, file_name="combined")
        remapped_combined_clusters = remap_clusters_to_labels(combined_df['NLOS'], combined_clusters)
        visualize_umap(combined_features, remapped_combined_clusters, combined_df['NLOS'], best_algorithm, output_dir,
        # visualize_umap(combined_features, combined_clusters, combined_df['NLOS'], best_algorithm, output_dir,
                       file_name="combined")
    except Exception as e:
        logging.error(f"[Combined] Failed to cluster or visualize: {e}")
        print(f"[Combined] Error: Failed to cluster or visualize: {e}")
        return

    summary_rows.append({
        'File': "Combined",
        'Algorithm': best_algorithm,
        'ARI': stats['metrics']['ARI'],
        'Silhouette': stats['metrics']['Silhouette'],
        'Completeness': stats['metrics']['Completeness'],
        'F1': stats['metrics']['F1'],
        'Accuracy': stats['metrics']['Accuracy'],
        'Precision': stats['metrics']['Precision'],
        'Recall': stats['metrics']['Recall'],
        'Clusters': stats['n_clusters'],
        'Noise Points': stats['noise_points']
    })

    try:
        summary = pd.DataFrame(summary_rows)
        print("\nSummary of Results:\n", summary)
        logging.info("\nSummary of Results:\n" + str(summary))
        summary.to_csv(os.path.join(output_dir, "clustering_summary.csv"))
        logging.info(f"Clustering summary saved to {os.path.join(output_dir, 'clustering_summary.csv')}")
    except Exception as e:
        logging.error(f"Failed to save clustering summary: {e}")
        print(f"Error: Failed to save clustering summary: {e}")


def main():
    pass


if __name__ == "__main__":
    # data is from the train results
    test_foler_name = "5"
    best_params = {'n_layers': 2, 'layer_sizes': [640, 768], 'latent_dim': 10, 'lr': 0.0026555062953972377, 'batch_size': 128, 'epochs': 350, 'patience': 90, 'best_algorithm': 'GMM', 'best_ari': 0.4350675213596458, 'best_params': {'n_components': 2}}
    trial = 62

    dataset_files = [
        "dataset/uwb_dataset_part1.csv",
        "dataset/uwb_dataset_part2.csv",
        "dataset/uwb_dataset_part3.csv",
        "dataset/uwb_dataset_part4.csv",
        "dataset/uwb_dataset_part5.csv",
        "dataset/uwb_dataset_part6.csv",
        "dataset/uwb_dataset_part7.csv"
    ]
    features = ["NLOS"] + [f"CIR{i}" for i in range(1016)] + ['RXPACC', 'RANGE', 'CIR_PWR', 'STDEV_NOISE', 'FP_AMP2',
                                                              'FP_AMP1', 'FP_IDX']
    cir_features = [f"CIR{i}" for i in range(1016)]
    selected_features = ['RXPACC', 'RANGE', 'CIR_PWR', 'STDEV_NOISE', 'FP_AMP2', 'FP_AMP1', 'FP_IDX']

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", f"test-{test_foler_name}-"+timestamp)
    print(f"Creating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {output_dir}")
    try:
        setup_logging(output_dir)
    except Exception as e:
        print(f"Error: Failed to initialize logging: {e}")
        raise

    model_path = os.path.join("outputs", test_foler_name, f"autoencoder_model_trial_{trial}.pth")
    cir_scaler_path = os.path.join("outputs", test_foler_name, "cir_scaler.pkl")
    selected_scaler_path = os.path.join("outputs", test_foler_name, "selected_scaler.pkl")

    best_clustering_params = best_params["best_params"]
    best_algorithm = best_params["best_algorithm"]

    try:
        test_on_individual_files(dataset_files, features, cir_features, selected_features, output_dir, best_algorithm,
                                 best_clustering_params, best_params, model_path, cir_scaler_path, selected_scaler_path)
    except Exception as e:
        logging.error(f"Main execution failed: {e}")
        print(f"Error: Main execution failed: {e}")