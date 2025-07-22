# UWB LOS/NLOS Clustering using Autoencoder + Unsupervised Learning

This project performs unsupervised clustering of Ultra-Wideband (UWB) channel impulse response (CIR) data to distinguish between LOS and NLOS conditions using an autoencoder and clustering algorithms.

## 📁 Dataset

The dataset used in this project is from:

**GitHub Repo**: [UWB LOS/NLOS Dataset by EWINE Project](https://github.com/ewine-project/UWB-LOS-NLOS-Data-Set)

### Download Instructions:

1. Clone or download the dataset:
   ```bash 
   git clone https://github.com/ewine-project/UWB-LOS-NLOS-Data-Set.git
2. Move the CSV files into a local dataset/ folder:
    ```bash 
   dataset/
   ├── uwb_dataset_part1.csv
   ├── uwb_dataset_part2.csv
   ├── ...
   └── uwb_dataset_part7.csv

The script expects the data in dataset/ relative to the project root.

## How to Run
1. Install Dependencies

Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
2. Train and Evaluate
To run the clustering evaluation script:









