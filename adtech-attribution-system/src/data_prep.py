"""
Data Preparation for AdTech Attribution
Generates realistic multi-touch user journey data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List
import json

class AttributionDataGenerator:
    """Generate synthetic attribution data"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.channels = ['google_search', 'facebook', 'instagram', 'tiktok', 'display', 'email', 'organic']
        self.device_types = ['mobile', 'desktop', 'tablet']
        self.countries = ['US', 'UK', 'CA', 'AU', 'DE']
        
    def generate_user_journeys(self, n_users: int = 10000, conversion_rate: float = 0.15) -> pd.DataFrame:
        """Generate user journey data with multiple touchpoints"""
        
        journeys = []
        
        for user_id in range(n_users):
            # Determine if user converts
            converts = np.random.random() < conversion_rate
            
            # Number of touchpoints (1-10, weighted towards 3-5)
            n_touchpoints = np.random.choice(
                range(1, 11), 
                p=[0.05, 0.15, 0.20, 0.20, 0.15, 0.10, 0.08, 0.04, 0.02, 0.01]
            )
            
            # Generate touchpoints for this user
            base_time = datetime.now() - timedelta(days=np.random.randint(1, 30))
            
            for touch_idx in range(n_touchpoints):
                # Add randomness to touchpoint timing
                touchpoint_time = base_time + timedelta(
                    hours=touch_idx * np.random.randint(2, 24),
                    minutes=np.random.randint(0, 60)
                )
                
                # Channel selection (influenced by conversion)
                if converts and touch_idx == n_touchpoints - 1:
                    # Last touchpoint more likely to be search if converts
                    channel = np.random.choice(
                        self.channels,
                        p=[0.40, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10]
                    )
                else:
                    channel = np.random.choice(self.channels)
                
                # Device type
                device = np.random.choice(self.device_types, p=[0.5, 0.4, 0.1])
                
                # Cost per touchpoint
                channel_costs = {
                    'google_search': 1.50,
                    'facebook': 0.80,
                    'instagram': 0.85,
                    'tiktok': 0.60,
                    'display': 0.40,
                    'email': 0.10,
                    'organic': 0.00
                }
                cost = channel_costs[channel]
                
                # Campaign info
                campaign_id = f"CAMP_{np.random.randint(1, 50):03d}"
                creative_id = f"CREA_{np.random.randint(1, 200):03d}"
                
                journey = {
                    'user_id': user_id,
                    'touchpoint_id': f"{user_id}_{touch_idx}",
                    'touchpoint_order': touch_idx,
                    'channel': channel,
                    'device': device,
                    'country': np.random.choice(self.countries),
                    'campaign_id': campaign_id,
                    'creative_id': creative_id,
                    'timestamp': touchpoint_time,
                    'cost': cost,
                    'converted': converts,
                    'conversion_value': 100.0 if converts else 0.0,
                    'time_since_first_touch_hours': touch_idx * 12  # Approximate
                }
                
                journeys.append(journey)
        
        df = pd.DataFrame(journeys)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
    
    def create_conversion_dataset(self, journeys_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create conversion-level dataset (one row per user journey)"""
        
        conversions = []
        
        for user_id in journeys_df['user_id'].unique():
            user_journey = journeys_df[journeys_df['user_id'] == user_id].sort_values('timestamp')
            
            if len(user_journey) == 0:
                continue
            
            converted = user_journey['converted'].iloc[0]
            
            if converted:
                conversion = {
                    'user_id': user_id,
                    'conversion_value': 100.0,
                    'conversion_timestamp': user_journey['timestamp'].iloc[-1],
                    'first_touch_channel': user_journey['channel'].iloc[0],
                    'last_touch_channel': user_journey['channel'].iloc[-1],
                    'num_touchpoints': len(user_journey),
                    'total_spend': user_journey['cost'].sum(),
                    'journey_duration_hours': (
                        user_journey['timestamp'].iloc[-1] - user_journey['timestamp'].iloc[0]
                    ).total_seconds() / 3600,
                    'channels_in_journey': ','.join(user_journey['channel'].unique()),
                    'touchpoints': user_journey.to_dict('records')
                }
                conversions.append(conversion)
        
        conversions_df = pd.DataFrame(conversions)
        
        # Separate touchpoints back out for training
        touchpoints = []
        for conv in conversions:
            for touch in conv['touchpoints']:
                touchpoints.append(touch)
        
        touchpoints_df = pd.DataFrame(touchpoints)
        
        return conversions_df, touchpoints_df


class FeatureEngineer:
    """Feature engineering for attribution models"""
    
    @staticmethod
    def extract_journey_features(journeys_df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from user journeys"""
        
        features = []
        
        for user_id in journeys_df['user_id'].unique():
            user_journey = journeys_df[journeys_df['user_id'] == user_id].sort_values('timestamp')
            
            if len(user_journey) == 0:
                continue
            
            # Channel features
            channel_counts = user_journey['channel'].value_counts().to_dict()
            
            # Device features
            device_counts = user_journey['device'].value_counts().to_dict()
            
            # Temporal features
            times = pd.to_datetime(user_journey['timestamp'])
            time_diffs = times.diff().dt.total_seconds() / 3600  # hours
            
            feature_row = {
                'user_id': user_id,
                'n_touchpoints': len(user_journey),
                'n_unique_channels': user_journey['channel'].nunique(),
                'n_unique_campaigns': user_journey['campaign_id'].nunique(),
                'total_cost': user_journey['cost'].sum(),
                'avg_time_between_touches_hours': time_diffs.mean(),
                'max_time_between_touches_hours': time_diffs.max(),
                'device_diversity': user_journey['device'].nunique(),
                'has_search': 1 if 'google_search' in user_journey['channel'].values else 0,
                'has_social': 1 if any(x in user_journey['channel'].values for x in ['facebook', 'instagram', 'tiktok']) else 0,
                'has_display': 1 if 'display' in user_journey['channel'].values else 0,
                'has_email': 1 if 'email' in user_journey['channel'].values else 0,
                'converted': user_journey['converted'].iloc[0],
                'conversion_value': user_journey['conversion_value'].iloc[0]
            }
            
            # Add channel-specific counts
            for channel in ['google_search', 'facebook', 'instagram', 'tiktok', 'display', 'email', 'organic']:
                feature_row[f'channel_{channel}_count'] = channel_counts.get(channel, 0)
            
            # Add device-specific counts
            for device in ['mobile', 'desktop', 'tablet']:
                feature_row[f'device_{device}_count'] = device_counts.get(device, 0)
            
            features.append(feature_row)
        
        return pd.DataFrame(features)


def prepare_training_data(n_users: int = 10000) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare complete training dataset
    Returns: journeys, conversions, features
    """
    
    print("🔄 Generating synthetic attribution data...")
    
    generator = AttributionDataGenerator(seed=42)
    journeys = generator.generate_user_journeys(n_users=n_users, conversion_rate=0.15)
    
    print(f"✓ Generated {len(journeys)} touchpoints for {n_users} users")
    
    # Create conversion dataset
    conversions, touchpoints = generator.create_conversion_dataset(journeys)
    
    print(f"✓ {len(conversions)} conversions recorded")
    print(f"✓ Average journey length: {journeys.groupby('user_id').size().mean():.2f} touchpoints")
    
    # Feature engineering
    features = FeatureEngineer.extract_journey_features(journeys)
    
    print(f"✓ Engineered {len(features.columns)} features")
    
    return journeys, conversions, features


if __name__ == "__main__":
    journeys, conversions, features = prepare_training_data(n_users=5000)
    
    print("\n📊 Dataset Summary:")
    print(f"Journeys shape: {journeys.shape}")
    print(f"Conversions shape: {conversions.shape}")
    print(f"Features shape: {features.shape}")
    
    print("\n📈 Conversion Statistics:")
    print(f"Conversion rate: {len(conversions) / len(journeys['user_id'].unique()) * 100:.2f}%")
    print(f"Avg conversion value: ${conversions['conversion_value'].mean():.2f}")
    print(f"Total revenue: ${conversions['conversion_value'].sum():.2f}")
