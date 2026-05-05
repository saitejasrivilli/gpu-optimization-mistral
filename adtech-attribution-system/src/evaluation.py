"""
Attribution Model Evaluation
Metrics for comparing attribution models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List


class AttributionEvaluator:
    """Evaluate and compare attribution models"""
    
    @staticmethod
    def channel_credit_summary(attribution_results_df: pd.DataFrame) -> pd.DataFrame:
        """Summarize credit by channel and model"""
        
        summary = attribution_results_df.groupby(['model', 'channel']).agg({
            'credit': ['sum', 'mean', 'count'],
            'credit_percentage': 'mean'
        }).round(2)
        
        summary.columns = ['total_credit', 'avg_credit', 'n_touches', 'avg_percentage']
        
        return summary
    
    @staticmethod
    def model_agreement(attribution_results_df: pd.DataFrame) -> Dict[str, float]:
        """Measure agreement between models"""
        
        # For each channel, calculate std dev of credit across models
        channel_std = attribution_results_df.groupby(['channel', 'model'])['credit_percentage'].mean().unstack(fill_value=0).std(axis=1)
        
        # Lower std = higher agreement
        agreement_score = 1 - (channel_std.mean() / 100)  # Normalize
        
        return {
            'overall_agreement': agreement_score,
            'channel_agreement': channel_std.to_dict()
        }
    
    @staticmethod
    def roi_potential(attribution_results_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate potential ROI improvements by reallocating budget"""
        
        roi_data = []
        
        for model in attribution_results_df['model'].unique():
            model_data = attribution_results_df[attribution_results_df['model'] == model]
            
            # Group by channel
            channel_credit = model_data.groupby('channel')['credit'].sum()
            channel_count = model_data.groupby('channel')['model'].count()
            
            # Assume spend follows current allocation
            channel_spend = model_data.groupby('channel')['model'].count() * 100  # Mock spend
            
            for channel in channel_credit.index:
                roi = (channel_credit[channel] / max(channel_spend[channel], 1)) * 100
                
                roi_data.append({
                    'model': model,
                    'channel': channel,
                    'total_credit': channel_credit[channel],
                    'touchpoints': channel_count[channel],
                    'credit_per_touch': channel_credit[channel] / channel_count[channel],
                    'roi_potential': roi
                })
        
        return pd.DataFrame(roi_data)
    
    @staticmethod
    def model_stability(attribution_results_df: pd.DataFrame) -> Dict[str, float]:
        """Measure model stability (consistency of credit allocation)"""
        
        stability_scores = {}
        
        for model in attribution_results_df['model'].unique():
            model_data = attribution_results_df[attribution_results_df['model'] == model]
            
            # Calculate coefficient of variation for each channel
            channel_cv = []
            for channel in model_data['channel'].unique():
                channel_data = model_data[model_data['channel'] == channel]['credit']
                if len(channel_data) > 1:
                    cv = channel_data.std() / channel_data.mean() if channel_data.mean() > 0 else 0
                    channel_cv.append(cv)
            
            # Lower CV = more stable
            stability = 1 - np.mean(channel_cv) if channel_cv else 1
            stability_scores[model] = max(0, stability)
        
        return stability_scores


class AttributionVisualizer:
    """Visualize attribution model comparisons"""
    
    @staticmethod
    def plot_channel_credit_comparison(attribution_results_df: pd.DataFrame, output_path: str = None):
        """Compare channel credit across models"""
        
        summary = attribution_results_df.groupby(['model', 'channel'])['credit'].sum().reset_index()
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        channels = summary['channel'].unique()
        x = np.arange(len(channels))
        width = 0.2
        
        models = summary['model'].unique()
        
        for i, model in enumerate(sorted(models)):
            model_data = summary[summary['model'] == model].sort_values('channel')
            ax.bar(x + i * width, model_data['credit'], width, label=model)
        
        ax.set_xlabel('Channel', fontsize=12, fontweight='bold')
        ax.set_ylabel('Total Credit ($)', fontsize=12, fontweight='bold')
        ax.set_title('Channel Credit Distribution by Attribution Model', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(channels, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {output_path}")
        
        return fig
    
    @staticmethod
    def plot_model_heatmap(attribution_results_df: pd.DataFrame, output_path: str = None):
        """Heatmap of channel credit by model"""
        
        pivot_data = attribution_results_df.groupby(['model', 'channel'])['credit'].sum().unstack(fill_value=0)
        
        # Normalize by total credit per model
        pivot_data = pivot_data.div(pivot_data.sum(axis=1), axis=0) * 100
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlGnBu', ax=ax, cbar_kws={'label': 'Credit %'})
        
        ax.set_title('Attribution Model Heatmap (% Credit by Channel)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Channel', fontsize=12, fontweight='bold')
        ax.set_ylabel('Model', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {output_path}")
        
        return fig
    
    @staticmethod
    def plot_model_agreement(agreement_data: Dict, output_path: str = None):
        """Visualize model agreement"""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        channels = list(agreement_data['channel_agreement'].keys())
        disagreement = [agreement_data['channel_agreement'][ch] for ch in channels]
        
        colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(channels)))
        
        ax.barh(channels, disagreement, color=colors)
        
        ax.set_xlabel('Disagreement Score (lower = higher agreement)', fontsize=12, fontweight='bold')
        ax.set_title(f'Model Agreement by Channel\n(Overall Agreement: {agreement_data["overall_agreement"]:.2%})', 
                     fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {output_path}")
        
        return fig
    
    @staticmethod
    def plot_roi_potential(roi_df: pd.DataFrame, output_path: str = None):
        """Visualize ROI potential by channel and model"""
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        pivot_data = roi_df.pivot(index='channel', columns='model', values='roi_potential')
        
        pivot_data.plot(kind='bar', ax=ax)
        
        ax.set_title('ROI Potential by Channel and Attribution Model', fontsize=14, fontweight='bold')
        ax.set_xlabel('Channel', fontsize=12, fontweight='bold')
        ax.set_ylabel('ROI Potential ($)', fontsize=12, fontweight='bold')
        ax.legend(title='Model')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {output_path}")
        
        return fig


if __name__ == "__main__":
    from models.simple_models import AttributionModelComparator
    from data_prep import prepare_training_data
    
    print("🔄 Preparing data...")
    journeys, conversions, features = prepare_training_data(n_users=1000)
    
    print("\n🔄 Running attribution comparison...")
    comparator = AttributionModelComparator()
    results = comparator.compare_models(journeys)
    
    print("\n📊 Evaluation Results:")
    print("\n1️⃣ Channel Credit Summary:")
    summary = AttributionEvaluator.channel_credit_summary(results)
    print(summary)
    
    print("\n2️⃣ Model Agreement:")
    agreement = AttributionEvaluator.model_agreement(results)
    print(f"Overall Agreement: {agreement['overall_agreement']:.2%}")
    
    print("\n3️⃣ Model Stability:")
    stability = AttributionEvaluator.model_stability(results)
    for model, score in stability.items():
        print(f"  {model}: {score:.2%}")
    
    print("\n4️⃣ ROI Potential:")
    roi = AttributionEvaluator.roi_potential(results)
    print(roi.groupby('model')['roi_potential'].mean())
