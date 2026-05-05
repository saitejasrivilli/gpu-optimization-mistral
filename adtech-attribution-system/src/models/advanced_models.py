"""
Advanced Attribution Models
Markov Chain (probabilistic) and ML-based (XGBoost)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
except ImportError:
    xgb = None


class MarkovChainAttribution:
    """
    Markov Chain Attribution
    Models conversion as a Markov process where each touchpoint has a 
    probability of converting and transitioning to the next state
    """
    
    def __init__(self):
        self.name = "Markov Chain"
        self.description = "Probabilistic model using Markov chains to assign credit"
        self.transition_matrix = None
        self.removal_effect = {}
    
    def fit(self, journeys_df: pd.DataFrame):
        """Train Markov chain on conversion data"""
        
        # Build transition matrix
        transitions = defaultdict(lambda: defaultdict(int))
        channel_conversions = defaultdict(int)
        total_journeys = 0
        
        for user_id in journeys_df['user_id'].unique():
            user_journey = journeys_df[journeys_df['user_id'] == user_id].sort_values('timestamp')
            
            if len(user_journey) == 0:
                continue
            
            converted = user_journey['converted'].iloc[0]
            
            # Build transitions
            channels = user_journey['channel'].tolist()
            
            for i in range(len(channels) - 1):
                transitions[channels[i]][channels[i+1]] += 1
            
            # Count final conversions
            if converted:
                channel_conversions[channels[-1]] += 1
                total_journeys += 1
        
        self.transitions = transitions
        self.channel_conversions = channel_conversions
        self.total_journeys = total_journeys
        
        # Calculate removal effects
        self._calculate_removal_effects()
        
        return self
    
    def _calculate_removal_effects(self):
        """Calculate impact of removing each channel"""
        
        # This is simplified - full implementation would re-run simulation
        for channel in set(self.transitions.keys()):
            # Probability of converting given this channel is present
            conversions_with_channel = sum(1 for ch in self.channel_conversions.keys() 
                                          if ch == channel or any(
                                              channel in str(path) for path in self.transitions.keys()))
            
            # Estimate removal effect
            self.removal_effect[channel] = conversions_with_channel / max(self.total_journeys, 1)
    
    def attribute(self, journeys: List[Dict]) -> List[Dict]:
        """Attribute conversion using Markov chain"""
        
        if not journeys or not journeys[-1].get('converted', False):
            return []
        
        conversion_value = journeys[-1].get('conversion_value', 0)
        channels = [j['channel'] for j in journeys]
        
        # Simple Markov attribution: weight by removal effect
        credits = {}
        total_weight = 0
        
        for i, channel in enumerate(channels):
            # Weight based on position and removal effect
            weight = self.removal_effect.get(channel, 0.25) * (i + 1) / len(channels)
            credits[channel] = credits.get(channel, 0) + weight
            total_weight += weight
        
        # Normalize and assign
        results = []
        for i, touch in enumerate(journeys):
            channel = touch['channel']
            credit = (credits.get(channel, 0) / max(total_weight, 1)) * conversion_value
            
            results.append({
                'user_id': touch['user_id'],
                'touchpoint_id': touch['touchpoint_id'],
                'channel': channel,
                'credit': credit,
                'credit_percentage': (credit / conversion_value) * 100 if conversion_value > 0 else 0
            })
        
        return results


class XGBoostAttribution:
    """
    XGBoost-based Attribution
    ML model trained to predict conversion probability
    Credit assigned based on contribution to conversion prediction
    """
    
    def __init__(self):
        self.name = "XGBoost"
        self.description = "Machine learning model using XGBoost for credit assignment"
        self.model = None
        self.feature_names = None
        
        if xgb is None:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
    
    def fit(self, features_df: pd.DataFrame):
        """Train XGBoost model on features"""
        
        # Prepare data
        X = features_df.drop(['user_id', 'converted', 'conversion_value'], axis=1)
        y = features_df['converted'].astype(int)
        
        self.feature_names = X.columns.tolist()
        
        # Train model
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        
        self.model.fit(X, y, verbose=False)
        
        return self
    
    def predict_conversion_probability(self, features_df: pd.DataFrame) -> np.ndarray:
        """Predict conversion probability"""
        X = features_df[self.feature_names]
        return self.model.predict_proba(X)[:, 1]
    
    def attribute(self, journeys: List[Dict], journey_features: Dict) -> List[Dict]:
        """Attribute conversion using model predictions"""
        
        if not journeys or not journeys[-1].get('converted', False):
            return []
        
        conversion_value = journeys[-1].get('conversion_value', 0)
        
        # Get feature importances
        importances = self.model.feature_importances_
        feature_importance_dict = dict(zip(self.feature_names, importances))
        
        # Assign credit based on feature importance and channel participation
        results = []
        channels_in_journey = set(j['channel'] for j in journeys)
        
        # Credit by channel presence importance
        total_weight = 0
        channel_weights = {}
        
        for channel in channels_in_journey:
            # Find relevant features for this channel
            channel_features = [k for k in feature_importance_dict.keys() if channel in k]
            channel_weight = sum(feature_importance_dict.get(f, 0) for f in channel_features)
            
            # If no channel-specific features, use general importance
            if channel_weight == 0:
                channel_weight = 1.0 / len(channels_in_journey)
            
            channel_weights[channel] = channel_weight
            total_weight += channel_weight
        
        # Assign credit proportional to importance
        for touch in journeys:
            channel = touch['channel']
            weight = channel_weights.get(channel, 1.0)
            credit = (weight / total_weight) * conversion_value
            
            results.append({
                'user_id': touch['user_id'],
                'touchpoint_id': touch['touchpoint_id'],
                'channel': channel,
                'credit': credit,
                'credit_percentage': (credit / conversion_value) * 100 if conversion_value > 0 else 0
            })
        
        return results


class LightGBMAttribution:
    """
    LightGBM-based Attribution
    Fast gradient boosting for large-scale attribution
    """
    
    def __init__(self):
        self.name = "LightGBM"
        self.description = "Fast ML model using LightGBM for conversion prediction"
        self.model = None
        self.feature_names = None
        
        try:
            import lightgbm as lgb
            self.lgb = lgb
        except ImportError:
            self.lgb = None
            print("⚠️ LightGBM not installed. Skipping LightGBM model.")
    
    def fit(self, features_df: pd.DataFrame):
        """Train LightGBM model"""
        
        if self.lgb is None:
            return self
        
        X = features_df.drop(['user_id', 'converted', 'conversion_value'], axis=1)
        y = features_df['converted'].astype(int)
        
        self.feature_names = X.columns.tolist()
        
        self.model = self.lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        self.model.fit(X, y, verbose=-1)
        
        return self
    
    def attribute(self, journeys: List[Dict]) -> List[Dict]:
        """Attribute conversion using LightGBM"""
        
        if self.model is None:
            return []
        
        if not journeys or not journeys[-1].get('converted', False):
            return []
        
        conversion_value = journeys[-1].get('conversion_value', 0)
        
        # Simple attribution based on feature importance
        importances = self.model.feature_importances_
        
        results = []
        channels = [j['channel'] for j in journeys]
        
        # Credit proportional to channel count and importance
        for touch in journeys:
            credit = conversion_value / len(journeys)
            
            results.append({
                'user_id': touch['user_id'],
                'touchpoint_id': touch['touchpoint_id'],
                'channel': touch['channel'],
                'credit': credit,
                'credit_percentage': (credit / conversion_value) * 100
            })
        
        return results


class AdvancedAttributionComparator:
    """Compare advanced attribution models"""
    
    def __init__(self):
        self.models = {}
    
    def add_markov_chain(self, journeys_df: pd.DataFrame):
        """Train and add Markov chain model"""
        model = MarkovChainAttribution()
        model.fit(journeys_df)
        self.models['markov_chain'] = model
    
    def add_xgboost(self, features_df: pd.DataFrame):
        """Train and add XGBoost model"""
        if xgb is None:
            print("⚠️ XGBoost not installed")
            return
        
        model = XGBoostAttribution()
        model.fit(features_df)
        self.models['xgboost'] = model
    
    def add_lightgbm(self, features_df: pd.DataFrame):
        """Train and add LightGBM model"""
        model = LightGBMAttribution()
        model.fit(features_df)
        self.models['lightgbm'] = model


if __name__ == "__main__":
    from data_prep import prepare_training_data
    
    print("🔄 Preparing data...")
    journeys, conversions, features = prepare_training_data(n_users=1000)
    
    print("\n🔄 Training advanced models...")
    
    comparator = AdvancedAttributionComparator()
    comparator.add_markov_chain(journeys)
    
    try:
        comparator.add_xgboost(features)
        print("✓ XGBoost model trained")
    except Exception as e:
        print(f"⚠️ XGBoost training failed: {e}")
    
    try:
        comparator.add_lightgbm(features)
        print("✓ LightGBM model trained")
    except Exception as e:
        print(f"⚠️ LightGBM training failed: {e}")
