import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score, completeness_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import umap
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
import logging
from datetime import datetime
import optuna
from sklearn.model_selection import train_test_split
import shutil

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
def load_datasets(dataset_files: list, features: list) -> pd.DataFrame:
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


# Train Autoencoder with Early Stopping
def train_autoencoder(model, data, output_dir, trial_number=None, epochs=200, batch_size=256, lr=0.001, patience=50):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    train_tensor = torch.tensor(train_data, dtype=torch.float32).to(device)
    val_tensor = torch.tensor(val_data, dtype=torch.float32).to(device)
    train_dataset = torch.utils.data.TensorDataset(train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               pin_memory=torch.cuda.is_available())

    best_loss = float('inf')
    patience_counter = 0
    model_path = os.path.join(output_dir,
                              f"autoencoder_model_trial_{trial_number}.pth" if trial_number is not None else "best_autoencoder_model.pth")
    losses = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            batch_data = batch[0].to(device)
            optimizer.zero_grad()
            recon, _ = model(batch_data)
            loss = criterion(recon, batch_data)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            recon, _ = model(val_tensor)
            val_loss = criterion(recon, val_tensor).item()

        losses.append((avg_train_loss, val_loss))
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            try:
                torch.save(model.state_dict(), model_path)
                logging.info(f"Saved model to {model_path}")
            except Exception as e:
                logging.error(f"Failed to save model to {model_path}: {e}")
                print(f"Error: Failed to save model to {model_path}: {e}")
                raise
        else:
            patience_counter += 1

        logging.info(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Patience: {patience}")
        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            logging.info(f"Early stopping at epoch {epoch + 1}")
            break

    try:
        model.load_state_dict(torch.load(model_path))
    except Exception as e:
        logging.error(f"Failed to load model from {model_path}: {e}")
        print(f"Error: Failed to load model from {model_path}: {e}")
        raise

    try:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(losses) + 1), [x[0] for x in losses], label='Train Loss')
        plt.plot(range(1, len(losses) + 1), [x[1] for x in losses], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Autoencoder Loss Over Epochs')
        plt.legend()
        plt.grid(True)
        loss_plot_path = os.path.join(output_dir,
                                      f"autoencoder_loss_{'trial_' + str(trial_number) if trial_number is not None else 'final'}.png")
        plt.savefig(loss_plot_path)
        plt.close()
        logging.info(f"Loss plot saved to {loss_plot_path}")
        print(f"Loss plot saved to {loss_plot_path}")
    except Exception as e:
        logging.error(f"Failed to save loss plot to {loss_plot_path}: {e}")
        print(f"Warning: Failed to save loss plot: {e}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model


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


# Objective Function for Optuna with Trial Logging
def objective(trial, data, selected_features_data, true_labels, output_dir):
    start_time = datetime.now()
    logging.info(f"Trial {trial.number}: Started at {start_time}")

    # Autoencoder hyperparameters
    n_layers = trial.suggest_int('n_layers', 2, 4)
    layer_sizes = [trial.suggest_int(f'layer_{i}', 128, 1024, step=128) for i in range(n_layers)]
    latent_dim = trial.suggest_int('latent_dim', 10, 50)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_int('batch_size', 32, 256, step=32)
    epochs = trial.suggest_int('epochs', 100, 500, step=50)
    patience = trial.suggest_int('patience', 20, 100, step=10)

    # Log trial parameters
    trial_params = {
        'n_layers': n_layers,
        'layer_sizes': layer_sizes,
        'latent_dim': latent_dim,
        'lr': lr,
        'batch_size': batch_size,
        'epochs': epochs,
        'patience': patience
    }
    logging.info(f"Trial {trial.number}: Parameters: {trial_params}")

    # Train autoencoder
    try:
        model = Autoencoder(input_dim=data.shape[1], layer_sizes=layer_sizes, latent_dim=latent_dim)
        model = train_autoencoder(model, data, output_dir, trial_number=trial.number, epochs=epochs,
                                  batch_size=batch_size, lr=lr, patience=patience)
    except Exception as e:
        logging.error(f"Trial {trial.number}: Autoencoder training failed: {e}")
        print(f"Error: Trial {trial.number}: Autoencoder training failed: {e}")
        raise

    # Get latent features
    try:
        latent_features = get_latent_features(model, data, batch_size=batch_size)
        combined_features = combine_features(latent_features, selected_features_data)
    except Exception as e:
        logging.error(f"Trial {trial.number}: Failed to get latent features: {e}")
        print(f"Error: Trial {trial.number}: Failed to get latent features: {e}")
        raise

    # Try different clustering algorithms
    best_ari = -1
    best_algorithm = None
    best_clusters = None
    best_params = None
    trial_metrics = []

    # K-means
    try:
        n_clusters = 2
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(combined_features)
        stats = analyze_clusters(clusters, true_labels, combined_features, 'KMeans', output_dir,
                                 params={'n_clusters': n_clusters}, file_name="combined")
        ari = stats['metrics']['ARI']
        f1 = stats['metrics']['F1']
        trial_metrics.append({'algorithm': 'KMeans', 'ARI': ari, 'F1': f1})
        if ari > best_ari:
            best_ari = ari
            best_algorithm = 'KMeans'
            best_clusters = clusters
            best_params = {'n_clusters': n_clusters}
    except Exception as e:
        logging.error(f"Trial {trial.number}: KMeans clustering failed: {e}")
        print(f"Error: Trial {trial.number}: KMeans clustering failed: {e}")

    # GMM
    try:
        n_components = 2
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        clusters = gmm.fit_predict(combined_features)
        stats = analyze_clusters(clusters, true_labels, combined_features, 'GMM', output_dir,
                                 params={'n_components': n_components}, file_name="combined")
        ari = stats['metrics']['ARI']
        f1 = stats['metrics']['F1']
        trial_metrics.append({'algorithm': 'GMM', 'ARI': ari, 'F1': f1})
        if ari > best_ari:
            best_ari = ari
            best_algorithm = 'GMM'
            best_clusters = clusters
            best_params = {'n_components': n_components}
    except Exception as e:
        logging.error(f"Trial {trial.number}: GMM clustering failed: {e}")
        print(f"Error: Trial {trial.number}: GMM clustering failed: {e}")

    # DBSCAN
    try:
        eps = trial.suggest_float('dbscan_eps', 0.1, 2.0)
        min_samples = trial.suggest_int('dbscan_min_samples', 5, 20)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(combined_features)
        if len(set(clusters) - {-1}) > 1:
            stats = analyze_clusters(clusters, true_labels, combined_features, 'DBSCAN', output_dir,
                                     params={'eps': eps, 'min_samples': min_samples}, file_name="combined")
            ari = stats['metrics']['ARI']
            f1 = stats['metrics']['F1']
            trial_metrics.append({'algorithm': 'DBSCAN', 'ARI': ari, 'F1': f1})
            if ari > best_ari:
                best_ari = ari
                best_algorithm = 'DBSCAN'
                best_clusters = clusters
                best_params = {'eps': eps, 'min_samples': min_samples}
    except Exception as e:
        logging.error(f"Trial {trial.number}: DBSCAN clustering failed: {e}")
        print(f"Error: Trial {trial.number}: DBSCAN clustering failed: {e}")

    # Log trial metrics
    for metric in trial_metrics:
        logging.info(f"Trial {trial.number}: {metric['algorithm']} - ARI: {metric['ARI']:.4f}, F1: {metric['F1']:.4f}")
    trial_params.update({'best_algorithm': best_algorithm, 'best_ari': best_ari, 'best_params': best_params})
    trial_metrics_df = pd.DataFrame(trial_metrics)
    trial_metrics_df['trial'] = trial.number
    trial_metrics_df['params'] = str(trial_params)
    try:
        trial_metrics_df.to_csv(os.path.join(output_dir, f"optuna_trial_{trial.number}.csv"), index=False)
        logging.info(
            f"Trial {trial.number}: Metrics saved to {os.path.join(output_dir, f'optuna_trial_{trial.number}.csv')}")
    except Exception as e:
        logging.error(f"Trial {trial.number}: Failed to save trial metrics: {e}")
        print(f"Error: Trial {trial.number}: Failed to save trial metrics: {e}")

    # Analyze best clustering
    if best_clusters is not None:
        try:
            stats = analyze_clusters(best_clusters, true_labels, combined_features, best_algorithm, output_dir,
                                     best_params, file_name="combined")
            trial.set_user_attr('F1', stats['metrics']['F1'])
            trial.set_user_attr('model_path', os.path.join(output_dir, f"autoencoder_model_trial_{trial.number}.pth"))
            trial.set_user_attr('best_algorithm', best_algorithm)
            trial.set_user_attr('best_clustering_params', best_params)
        except Exception as e:
            logging.error(f"Trial {trial.number}: Failed to analyze best clustering: {e}")
            print(f"Error: Trial {trial.number}: Failed to analyze best clustering: {e}")
    else:
        logging.error(f"Trial {trial.number}: No valid clustering results obtained")
        print(f"Error: Trial {trial.number}: No valid clustering results obtained")
        best_ari = -1

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logging.info(f"Trial {trial.number}: Ended at {end_time}, Duration: {duration:.2f} seconds")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_ari


# Plot Optuna Metrics
def plot_optuna_metrics(study, output_dir, suffix=""):
    try:
        trials = study.trials
        trial_data = []
        for trial in trials:
            if trial.value is not None:
                trial_data.append({
                    'trial': trial.number,
                    'ARI': trial.value,
                    'F1': trial.user_attrs.get('F1', 0)
                })
        df = pd.DataFrame(trial_data)
        if not df.empty:
            plt.figure(figsize=(10, 6))
            plt.plot(df['trial'], df['ARI'], label='ARI', marker='o')
            plt.plot(df['trial'], df['F1'], label='F1', marker='s')
            plt.xlabel('Trial Number')
            plt.ylabel('Score')
            plt.title(f'Optuna Optimization: ARI and F1 Scores{suffix}')
            plt.legend()
            plt.grid(True)
            plot_path = os.path.join(output_dir, f"optuna_metrics{suffix}.png")
            plt.savefig(plot_path)
            plt.close()
            logging.info(f"Optuna metrics plot saved to {plot_path}")
            print(f"Optuna metrics plot saved to {plot_path}")
            df.to_csv(os.path.join(output_dir, f"optuna_trials{suffix}.csv"), index=False)
            logging.info(f"Optuna trials saved to {os.path.join(output_dir, f'optuna_trials{suffix}.csv')}")
        else:
            logging.warning(f"No valid trials to plot for {suffix}")
            print(f"Warning: No valid trials to plot for {suffix}")
    except Exception as e:
        logging.error(f"Failed to generate Optuna metrics plot for {suffix}: {e}")
        print(f"Warning: Failed to generate Optuna metrics plot for {suffix}: {e}")


# Main Function
def main(dataset_files, features, cir_features, selected_features):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", timestamp)
    print(f"Creating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {output_dir}")

    print(f"Starting dataset loading...")
    try:
        df = load_datasets(dataset_files, features)
    except Exception as e:
        logging.error(f"Failed to load datasets: {e}")
        print(f"Error: Failed to load datasets: {e}")
        raise

    selected_cir_features = cir_features
    # all_features = ["NLOS"] + selected_cir_features + selected_features

    try:
        scaled_cir_features, cir_scaler = normalize_features(df, selected_cir_features,
                                                             scaler_path=os.path.join(output_dir, "cir_scaler.pkl"))
        scaled_selected_features, selected_scaler = normalize_features(df, selected_features,
                                                                       scaler_path=os.path.join(output_dir,
                                                                                                "selected_scaler.pkl"))
    except Exception as e:
        logging.error(f"Failed to normalize features: {e}")
        print(f"Error: Failed to normalize features: {e}")
        raise

    study = optuna.create_study(direction='maximize')
    try:
        study.optimize(
            lambda trial: objective(trial, scaled_cir_features, scaled_selected_features, df['NLOS'], output_dir),
            n_trials=100)
    except Exception as e:
        logging.error(f"Optuna optimization failed: {e}")
        print(f"Error: Optuna optimization failed: {e}")
        raise

    try:
        plot_optuna_metrics(study, output_dir)
    except Exception as e:
        logging.error(f"Failed to plot Optuna metrics: {e}")
        print(f"Error: Failed to plot Optuna metrics: {e}")

    # Save the best model
    best_trial = study.best_trial
    best_params = best_trial.params
    best_model_path = best_trial.user_attrs.get('model_path', None)
    best_algorithm = best_trial.user_attrs.get('best_algorithm', None)
    best_clustering_params = best_trial.user_attrs.get('best_clustering_params', None)
    if best_model_path and os.path.exists(best_model_path):
        try:
            shutil.copy(best_model_path, os.path.join(output_dir, "best_autoencoder_model.pth"))
            logging.info(
                f"Saved best model from trial {best_trial.number} to {os.path.join(output_dir, 'best_autoencoder_model.pth')}")
        except Exception as e:
            logging.error(f"Failed to save best model: {e}")
            print(f"Error: Failed to save best model: {e}")
            raise
    else:
        logging.error(f"Best model path not found for trial {best_trial.number}")
        print(f"Error: Best model path not found for trial {best_trial.number}")
        raise FileNotFoundError(f"Best model path not found for trial {best_trial.number}")

    logging.info(f"Best parameters: {best_params}")
    logging.info(f"Best clustering algorithm: {best_algorithm}")
    logging.info(f"Best clustering parameters: {best_clustering_params}")
    print(f"Best parameters: {best_params}")
    print(f"Best clustering algorithm: {best_algorithm}")
    print(f"Best clustering parameters: {best_clustering_params}")

    try:
        logging.info("Retraining final model on full dataset with best parameters")
        layer_sizes = [best_params[f'layer_{i}'] for i in range(best_params['n_layers'])]
        model = Autoencoder(input_dim=len(selected_cir_features), layer_sizes=layer_sizes,
                            latent_dim=best_params['latent_dim'])
        model.load_state_dict(torch.load(os.path.join(output_dir, "best_autoencoder_model.pth")))
        model = model.to(device)
        model = train_autoencoder(model, scaled_cir_features, output_dir,
                                  trial_number=None, epochs=best_params['epochs'], batch_size=best_params['batch_size'],
                                  lr=best_params['lr'], patience=best_params['patience'])
    except Exception as e:
        logging.error(f"Failed to train final model: {e}")
        print(f"Error: Failed to train final model: {e}")
        raise

    try:
        latent_features = get_latent_features(model, scaled_cir_features, batch_size=best_params['batch_size'])
        combined_features = combine_features(latent_features, scaled_selected_features)
    except Exception as e:
        logging.error(f"Failed to get latent features for final model: {e}")
        print(f"Error: Failed to get latent features for final model: {e}")
        raise

    try:
        if best_algorithm == 'KMeans':
            clusterer = KMeans(n_clusters=best_clustering_params['n_clusters'], random_state=42)
        elif best_algorithm == 'GMM':
            clusterer = GaussianMixture(n_components=best_clustering_params['n_components'], random_state=42)
        else:
            clusterer = DBSCAN(eps=best_clustering_params['eps'], min_samples=best_clustering_params['min_samples'])
        clusters = clusterer.fit_predict(combined_features)
        stats = analyze_clusters(clusters, df['NLOS'], combined_features, best_algorithm, output_dir,
                                 best_clustering_params, file_name="combined")
    except Exception as e:
        logging.error(f"Failed to cluster or visualize final model: {e}")
        print(f"Error: Failed to cluster or visualize final model: {e}")
        raise

    metrics_str = ", ".join([f"{k}={v:.3f}" for k, v in stats['metrics'].items()])
    print(f"\nBest Algorithm: {best_algorithm}")
    print(f"Parameters: {best_clustering_params}")
    print(f"Metrics: {metrics_str}, Clusters={stats['n_clusters']}, Noise Points={stats['noise_points']}")
    print(f"Cluster Sizes: {stats['cluster_sizes']}")
    print(f"Cluster Proportions: {stats['cluster_proportions']}")
    logging.info(f"Best Algorithm: {best_algorithm}")
    logging.info(f"Parameters: {best_clustering_params}")
    logging.info(f"Metrics: {metrics_str}, Clusters={stats['n_clusters']}, Noise Points={stats['noise_points']}")
    logging.info(f"Cluster Sizes: {stats['cluster_sizes']}")
    logging.info(f"Cluster Proportions: {stats['cluster_proportions']}")

    return output_dir, timestamp, best_algorithm, best_clustering_params, best_params


if __name__ == "__main__":
    dataset_files = [
        "dataset/uwb_dataset_part5.csv" # Kitchen w/ Living Room
    ]
    features = ["NLOS"] + [f"CIR{i}" for i in range(1016)] + ['RXPACC', 'RANGE', 'CIR_PWR', 'STDEV_NOISE', 'FP_AMP2',
                                                              'FP_AMP1', 'FP_IDX']
    cir_features = [f"CIR{i}" for i in range(1016)]
    selected_features = ['RXPACC', 'RANGE', 'CIR_PWR', 'STDEV_NOISE', 'FP_AMP2', 'FP_AMP1', 'FP_IDX']

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", timestamp)
    print(f"Creating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {output_dir}")
    try:
        setup_logging(output_dir)
    except Exception as e:
        print(f"Error: Failed to initialize logging: {e}")
        raise

    try:
        output_dir, timestamp, best_algorithm, best_clustering_params, best_params = main(dataset_files,
                                                                                          features,
                                                                                          cir_features,
                                                                                          selected_features)
    except Exception as e:
        logging.error(f"Main execution failed: {e}")
        print(f"Error: Main execution failed: {e}")