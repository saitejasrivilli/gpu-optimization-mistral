"""
Simple Attribution Models
Last-click, Linear, Time-decay, Position-based
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class AttributionResult:
    """Attribution result for a single journey"""
    user_id: str
    touchpoint_id: str
    channel: str
    credit: float
    credit_percentage: float


class LastClickAttribution:
    """Last-click attribution - 100% credit to final touchpoint"""
    
    def __init__(self):
        self.name = "Last-Click"
        self.description = "100% credit to the last touchpoint before conversion"
    
    def attribute(self, journeys: List[Dict]) -> List[AttributionResult]:
        """Attribute conversion to last touchpoint"""
        results = []
        
        if not journeys or not journeys[-1].get('converted', False):
            return results
        
        conversion_value = journeys[-1].get('conversion_value', 0)
        last_touch = journeys[-1]
        
        results.append(AttributionResult(
            user_id=last_touch['user_id'],
            touchpoint_id=last_touch['touchpoint_id'],
            channel=last_touch['channel'],
            credit=conversion_value,
            credit_percentage=100.0
        ))
        
        return results


class LinearAttribution:
    """Linear attribution - Equal credit to all touchpoints"""
    
    def __init__(self):
        self.name = "Linear"
        self.description = "Equal credit distributed across all touchpoints"
    
    def attribute(self, journeys: List[Dict]) -> List[AttributionResult]:
        """Distribute credit equally"""
        results = []
        
        if not journeys or not journeys[-1].get('converted', False):
            return results
        
        conversion_value = journeys[-1].get('conversion_value', 0)
        n_touches = len(journeys)
        credit_per_touch = conversion_value / n_touches
        
        for touch in journeys:
            results.append(AttributionResult(
                user_id=touch['user_id'],
                touchpoint_id=touch['touchpoint_id'],
                channel=touch['channel'],
                credit=credit_per_touch,
                credit_percentage=100.0 / n_touches
            ))
        
        return results


class TimeDecayAttribution:
    """Time-decay attribution - Recent touchpoints weighted more"""
    
    def __init__(self, decay_rate: float = 0.5):
        """
        decay_rate: How quickly weight decays (0-1)
        0.5 = half-life model (each touchpoint has 50% weight of previous)
        """
        self.decay_rate = decay_rate
        self.name = "Time-Decay"
        self.description = f"Recent touchpoints weighted {decay_rate*100:.0f}% more than earlier ones"
    
    def attribute(self, journeys: List[Dict]) -> List[AttributionResult]:
        """Attribute with time decay weighting"""
        results = []
        
        if not journeys or not journeys[-1].get('converted', False):
            return results
        
        conversion_value = journeys[-1].get('conversion_value', 0)
        n_touches = len(journeys)
        
        # Calculate weights (exponential decay from first to last)
        weights = []
        for i in range(n_touches):
            # Weight increases as we move toward end
            weight = self.decay_rate ** (n_touches - 1 - i)
            weights.append(weight)
        
        total_weight = sum(weights)
        
        for i, touch in enumerate(journeys):
            credit = (weights[i] / total_weight) * conversion_value
            credit_pct = (weights[i] / total_weight) * 100
            
            results.append(AttributionResult(
                user_id=touch['user_id'],
                touchpoint_id=touch['touchpoint_id'],
                channel=touch['channel'],
                credit=credit,
                credit_percentage=credit_pct
            ))
        
        return results


class PositionBasedAttribution:
    """Position-based attribution - First and last weighted more"""
    
    def __init__(self, first_weight: float = 0.4, last_weight: float = 0.4):
        """
        40% to first, 40% to last, 20% distributed to middle touches
        """
        self.first_weight = first_weight
        self.last_weight = last_weight
        self.middle_weight = 1.0 - first_weight - last_weight
        self.name = "Position-Based"
        self.description = f"First {first_weight*100:.0f}% + Last {last_weight*100:.0f}% + Middle {self.middle_weight*100:.0f}%"
    
    def attribute(self, journeys: List[Dict]) -> List[AttributionResult]:
        """Attribute based on position"""
        results = []
        
        if not journeys or not journeys[-1].get('converted', False):
            return results
        
        conversion_value = journeys[-1].get('conversion_value', 0)
        n_touches = len(journeys)
        
        if n_touches == 1:
            # Single touchpoint gets 100%
            results.append(AttributionResult(
                user_id=journeys[0]['user_id'],
                touchpoint_id=journeys[0]['touchpoint_id'],
                channel=journeys[0]['channel'],
                credit=conversion_value,
                credit_percentage=100.0
            ))
        
        elif n_touches == 2:
            # Two touches: split between first and last
            first_credit = conversion_value * self.first_weight
            last_credit = conversion_value * self.last_weight
            remaining_credit = conversion_value * self.middle_weight
            
            # Distribute remaining to first and last equally
            first_credit += remaining_credit / 2
            last_credit += remaining_credit / 2
            
            results.append(AttributionResult(
                user_id=journeys[0]['user_id'],
                touchpoint_id=journeys[0]['touchpoint_id'],
                channel=journeys[0]['channel'],
                credit=first_credit,
                credit_percentage=(first_credit / conversion_value) * 100
            ))
            results.append(AttributionResult(
                user_id=journeys[1]['user_id'],
                touchpoint_id=journeys[1]['touchpoint_id'],
                channel=journeys[1]['channel'],
                credit=last_credit,
                credit_percentage=(last_credit / conversion_value) * 100
            ))
        
        else:
            # 3+ touches: allocate first, last, and middle
            first_credit = conversion_value * self.first_weight
            last_credit = conversion_value * self.last_weight
            middle_total = conversion_value * self.middle_weight
            middle_per_touch = middle_total / (n_touches - 2)
            
            for i, touch in enumerate(journeys):
                if i == 0:
                    credit = first_credit
                    credit_pct = (first_credit / conversion_value) * 100
                elif i == n_touches - 1:
                    credit = last_credit
                    credit_pct = (last_credit / conversion_value) * 100
                else:
                    credit = middle_per_touch
                    credit_pct = (middle_per_touch / conversion_value) * 100
                
                results.append(AttributionResult(
                    user_id=touch['user_id'],
                    touchpoint_id=touch['touchpoint_id'],
                    channel=touch['channel'],
                    credit=credit,
                    credit_percentage=credit_pct
                ))
        
        return results


class AttributionModelComparator:
    """Compare multiple attribution models"""
    
    def __init__(self):
        self.models = {
            'last_click': LastClickAttribution(),
            'linear': LinearAttribution(),
            'time_decay': TimeDecayAttribution(decay_rate=0.5),
            'position_based': PositionBasedAttribution(first_weight=0.4, last_weight=0.4)
        }
    
    def compare_models(self, journeys_df: pd.DataFrame) -> pd.DataFrame:
        """Compare all models on same dataset"""
        
        results_list = []
        
        for user_id in journeys_df['user_id'].unique():
            user_journey = journeys_df[journeys_df['user_id'] == user_id].sort_values('timestamp')
            
            if len(user_journey) == 0 or not user_journey['converted'].iloc[0]:
                continue
            
            journey_dicts = user_journey.to_dict('records')
            
            # Apply each model
            for model_name, model in self.models.items():
                attributions = model.attribute(journey_dicts)
                
                for attr in attributions:
                    results_list.append({
                        'user_id': attr.user_id,
                        'model': model_name,
                        'channel': attr.channel,
                        'credit': attr.credit,
                        'credit_percentage': attr.credit_percentage,
                        'num_touches': len(journey_dicts)
                    })
        
        return pd.DataFrame(results_list)


if __name__ == "__main__":
    from data_prep import prepare_training_data
    
    print("🔄 Preparing data...")
    journeys, conversions, features = prepare_training_data(n_users=1000)
    
    print("\n🔄 Comparing attribution models...")
    comparator = AttributionModelComparator()
    comparison_results = comparator.compare_models(journeys)
    
    print("\n📊 Channel Credit by Model:")
    channel_credit = comparison_results.groupby(['model', 'channel'])['credit'].sum().unstack(fill_value=0)
    print(channel_credit)
    
    print("\n💰 Total Credit Distributed:")
    print(comparison_results.groupby('model')['credit'].sum())
