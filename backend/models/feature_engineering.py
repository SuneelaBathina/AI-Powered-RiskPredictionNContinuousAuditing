import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os

# Set higher log level for this module to reduce spam
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)  # Only show warnings and errors

class FeatureEngineer:
    """Feature engineering for financial transactions"""
    
    def __init__(self):
        self.feature_stats = {}
        self._batch_mode = False
        
    def create_features(self, df, batch_mode=False):
        """Create features for risk prediction"""
        df = df.copy()
        
        # Only log for batches, not single transactions
        if batch_mode or len(df) >= 100:
            logger.info(f"Creating features for {len(df)} transactions")
        
        # Time-based features
        df = self._create_time_features(df)
        
        # Amount-based features
        df = self._create_amount_features(df)
        
        # Transaction pattern features
        df = self._create_pattern_features_fixed(df)
        
        # Velocity features
        df = self._create_velocity_features_fixed(df)
        
        # Location-based features
        df = self._create_location_features(df)
        
        # Customer history features
        df = self._create_customer_features_fixed(df)
        
        # Interaction features
        df = self._create_interaction_features(df)
        
        # Fill any remaining NaN values
        df = df.fillna(0)
        
        # Only log summary for batches
        if batch_mode or len(df) >= 100:
            logger.info(f"Feature creation complete. Total features: {len(df.columns)}")
        
        return df
    
    # Rest of the methods remain the same...