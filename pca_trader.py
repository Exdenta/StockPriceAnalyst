
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


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from scipy.optimize import minimize
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class PCATrader:
    def __init__(self, sectors_dict, lookback_period=252):
        self.sectors = sectors_dict
        self.lookback_period = lookback_period
        self.all_stocks = [stock for stocks in sectors_dict.values() for stock in stocks]
        self.pca = PCA(n_components=3)
        self.scaler = StandardScaler()
        
    def fetch_data(self, start_date, end_date):
        """Fetch stock data and handle missing values"""
        self.data = yf.download(self.all_stocks, start=start_date, end=end_date)['Close']
        
        # Remove stocks with too many missing values
        missing_pct = self.data.isnull().mean()
        valid_stocks = missing_pct[missing_pct < 0.1].index
        self.data = self.data[valid_stocks]
        
        # Forward fill remaining missing values
        self.data = self.data.fillna(method='ffill')
        
        # Calculate percentage returns
        self.returns = self.data.pct_change() # Daily returns
        self.returns = self.returns.dropna() # Remove NA values
        
        print(f"Using {len(valid_stocks)} stocks after removing those with missing data")
        return self.data
        
    def calculate_pca_components(self):
        """Calculate PCA components from return data"""
        if len(self.returns) == 0:
            raise ValueError("No return data available")
            
        scaled_returns = self.scaler.fit_transform(self.returns)
        self.pca_components = self.pca.fit_transform(scaled_returns)
        self.loadings = pd.DataFrame(
            self.pca.components_.T,
            columns=['PC1', 'PC2', 'PC3'],
            index=self.returns.columns
        )
        
        # Print explained variance
        print("\nExplained Variance Ratio:")
        for i, var in enumerate(self.pca.explained_variance_ratio_, 1):
            print(f"PC{i}: {var:.2%}")
            
        return self.pca_components
        
    def generate_signals(self, threshold=1.0):
        """Generate trading signals based on PCA components"""
        signals = pd.DataFrame(index=self.returns.index)
        
        # Market signal (PC1)
        signals['market'] = np.where(self.pca_components[:, 0] > threshold, 1, 
                                   np.where(self.pca_components[:, 0] < -threshold, -1, 0))
        
        # Sector rotation signal (PC2)
        signals['sector'] = np.where(self.pca_components[:, 1] > threshold, 1,
                                   np.where(self.pca_components[:, 1] < -threshold, -1, 0))
        
        return signals

# Portfolio Management Class
class PortfolioManager:
    def __init__(self, pca_trader):
        self.pca_trader = pca_trader
        
    def optimize_portfolio(self, target_risk=0.15, risk_free_rate=0.02):
        """Optimize portfolio based on PCA loadings"""
        # Use only valid stocks
        self.valid_stocks = self.pca_trader.returns.columns
        
        def objective(weights):
            return -self._calculate_sharpe_ratio(weights, risk_free_rate)
            
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
            {'type': 'ineq', 'fun': lambda x: target_risk - self._calculate_portfolio_risk(x)}
        ]
        
        # Use number of valid stocks for initialization
        n_stocks = len(self.valid_stocks)
        bounds = tuple((0, 0.2) for _ in range(n_stocks))
        
        result = minimize(objective, 
                        x0=np.array([1/n_stocks]*n_stocks),
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints)
                        
        # Return series with valid stocks only
        return pd.Series(result.x, index=self.valid_stocks)
        
    def _calculate_portfolio_risk(self, weights):
        """Calculate portfolio risk based on PCA components"""
        return np.sqrt(np.dot(weights.T, np.dot(self.pca_trader.returns.cov() * 252, weights)))
    
    def _calculate_sharpe_ratio(self, weights, risk_free_rate):
        """Calculate portfolio Sharpe ratio"""
        returns = np.sum(self.pca_trader.returns.mean() * weights) * 252
        risk = self._calculate_portfolio_risk(weights)
        return (returns - risk_free_rate) / risk if risk > 0 else -np.inf

    def get_portfolio_statistics(self, weights):
        """Get detailed portfolio statistics"""
        stats = {
            'Annual Return': np.sum(self.pca_trader.returns.mean() * weights) * 252,
            'Annual Risk': self._calculate_portfolio_risk(weights),
            'Sharpe Ratio': self._calculate_sharpe_ratio(weights, 0.02),
            'Sector Allocation': self._calculate_sector_allocation(weights)
        }
        return stats
    
    def _calculate_sector_allocation(self, weights):
        """Calculate allocation by sector"""
        sector_alloc = {}
        weights_series = pd.Series(weights, index=self.valid_stocks)
        
        for sector, stocks in self.pca_trader.sectors.items():
            # Only consider stocks that are in our valid set
            valid_sector_stocks = [s for s in stocks if s in self.valid_stocks]
            sector_alloc[sector] = weights_series[valid_sector_stocks].sum()
            
        return pd.Series(sector_alloc)

def main():
    # Define sectors and stocks
    sectors = {
        'Technology': [
            'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'INTC', 'AMD', 'CRM', 'ADBE', 'ORCL', 'CSCO',
            'QCOM', 'TXN', 'IBM', 'NOW', 'AMAT', 'ADI', 'MU', 'LRCX', 'KLAC', 'SNPS'
        ],
        'Consumer': [
            'AMZN', 'TSLA', 'WMT', 'HD', 'NKE', 'SBUX', 'MCD', 'PG', 'KO', 'PEP',
            'COST', 'TGT', 'LOW', 'DG', 'DLTR', 'EL', 'CL', 'KMB', 'GIS', 'K'
        ],
        'Financial': [
            'JPM', 'GS', 'V', 'MA', 'BAC', 'WFC', 'MS', 'BLK', 'C', 'AXP',
            'SCHW', 'USB', 'PNC', 'TFC', 'COF', 'ALL', 'MMC', 'AON', 'MET', 'PRU'
        ],
        'Healthcare': [
            'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT', 'DHR', 'BMY',
            'AMGN', 'CVS', 'CI', 'ISRG', 'GILD', 'REGN', 'VRTX', 'HUM', 'BSX', 'ZTS'
        ],
        'Energy': [
            'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'OXY', 'PSX', 'VLO', 'MPC',
            'KMI', 'WMB', 'DVN', 'HAL', 'BKR', 'APA', 'HES', 'FANG', 'MRO', 'CVI'
        ],
        'Industrial': [
            'GE', 'HON', 'UPS', 'BA', 'CAT', 'DE', 'MMM', 'LMT', 'RTX', 'UNP',
            'FDX', 'NSC', 'EMR', 'ETN', 'GD', 'NOC', 'WM', 'RSG', 'PH', 'ROK'
        ],
        'Materials': [
            'LIN', 'SHW', 'APD', 'FCX', 'NEM', 'DOW', 'ECL', 'PPG', 'NUE', 'VMC',
            'ALB', 'CTVA', 'DD', 'FMC', 'CF', 'MOS', 'CE', 'PKG', 'AVY', 'BLL'
        ],
        'Real Estate': [
            'PLD', 'AMT', 'EQIX', 'CCI', 'PSA', 'WELL', 'SPG', 'DLR', 'O', 'VICI',
            'AVB', 'EQR', 'MAA', 'ESS', 'BXP', 'ARE', 'UDR', 'VTR', 'KIM', 'REG'
        ]
    }

    # Initialize PCA Trader with error handling
    try:
        trader = PCATrader(sectors)
        data = trader.fetch_data('2022-01-01', '2025-02-24')
        
        if data is not None and not data.empty:
            components = trader.calculate_pca_components()
            signals = trader.generate_signals(threshold=1.0)
            
            # Initialize Portfolio Manager and optimize
            portfolio = PortfolioManager(trader)
            optimal_weights = portfolio.optimize_portfolio()
            
            # Get and print portfolio statistics
            stats = portfolio.get_portfolio_statistics(optimal_weights)
            
            print("\nPortfolio Statistics:")
            print(f"Expected Annual Return: {stats['Annual Return']:.2%}")
            print(f"Annual Risk: {stats['Annual Risk']:.2%}")
            print(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
            
            print("\nTop 10 Stock Allocations:")
            print(optimal_weights.sort_values(ascending=False).head(10))
            
            print("\nSector Allocation:")
            print(stats['Sector Allocation'].sort_values(ascending=False))
            
            print("\nTrading Signals (last 5 days):")
            print(signals.tail())
            
        else:
            print("Error: No data retrieved")
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()
