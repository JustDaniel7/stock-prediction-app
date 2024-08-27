import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

PCA_COMPONENTS = 7  # Number of principal components
SEQUENCE_LENGTH = 60


def scale_and_reduce(df, pca_components=PCA_COMPONENTS):
    features = df.drop(['Date', 'Close'], axis=1)

    # Scale the features
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    # Apply PCA for feature reduction
    pca = PCA(n_components=pca_components)
    reduced_features = pca.fit_transform(scaled_features)

    return reduced_features, df['Close'].values


def create_sequences(data, targets, sequence_length=SEQUENCE_LENGTH):
    sequences, labels = [], []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
        labels.append(targets[i + sequence_length])
    return np.array(sequences), np.array(labels)
