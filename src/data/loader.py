"""Data loading utilities."""
import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Handle data loading operations."""
    
    @staticmethod
    def load_csv(file_path: Path) -> pd.DataFrame:
        """Load CSV file into DataFrame."""
        try:
            logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    @staticmethod
    def get_data_info(df: pd.DataFrame) -> dict:
        """Get basic information about the dataset."""
        info = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicates": df.duplicated().sum()
        }
        return info
