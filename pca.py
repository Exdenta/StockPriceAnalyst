
'''
Principal Component Analysis (PCA) - Explained
What is PCA?
Principal Component Analysis (PCA) is a dimensionality reduction technique used in machine learning and statistics. It transforms a dataset into a smaller number of uncorrelated features (called principal components) while preserving the most important information.

Why Use PCA?
Reduce dimensionality: High-dimensional data can be hard to visualize and process efficiently.
Remove redundancy: PCA removes correlated features, keeping only the essential information.
Improve performance: Reducing dimensions can speed up machine learning models and prevent overfitting.
Data visualization: PCA can project high-dimensional data into 2D or 3D for visualization.
'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import yfinance as yf


def main():
    # Step 1: Load stock data (e.g., S&P 500 stocks)
    print("1: Load stock data (e.g., S&P 500 stocks)")
    stocks = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "JPM", "GS", "V", "NFLX", "META"]
    data = yf.download(stocks, start="2020-01-01", end="2023-01-01")["Adj Close"]

    # Step 2: Standardize the stock price data
    print("2. Standardize the stock price data")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Step 3: Apply PCA
    pca = PCA(n_components=3)  # Keep top 3 components
    data_pca = pca.fit_transform(data_scaled)

    # Step 4: Explained variance
    print("Explained Variance Ratio:", pca.explained_variance_ratio_)

    # Step 5: Visualize cumulative variance
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Explained Variance vs. Number of Components")
    plt.show()


if __name__ == "__main__":
    main()
