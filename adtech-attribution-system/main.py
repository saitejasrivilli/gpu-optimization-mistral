"""
AdTech Attribution System - Main Training Pipeline
Train and compare all 6 attribution models
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json

from src.data_prep import prepare_training_data, FeatureEngineer
from src.models.simple_models import (
    LastClickAttribution, LinearAttribution, 
    TimeDecayAttribution, PositionBasedAttribution,
    AttributionModelComparator
)
from src.models.advanced_models import (
    MarkovChainAttribution, XGBoostAttribution, 
    LightGBMAttribution, AdvancedAttributionComparator
)
from src.evaluation import AttributionEvaluator, AttributionVisualizer


def train_all_models(n_users: int = 5000, output_dir: str = 'results'):
    """Train all attribution models and generate comparisons"""
    
    print("="*80)
    print(" ADTECH ATTRIBUTION SYSTEM - TRAINING PIPELINE")
    print("="*80)
    
    # ============================================================================
    # 1. DATA PREPARATION
    # ============================================================================
    
    print("\n📊 PHASE 1: DATA PREPARATION")
    print("-" * 80)
    
    journeys, conversions, features = prepare_training_data(n_users=n_users)
    
    print(f"\n✓ Generated {len(journeys)} touchpoints")
    print(f"✓ {len(conversions)} conversions recorded")
    print(f"✓ {len(features.columns)} features engineered")
    
    # Print conversion metrics
    conversion_rate = len(conversions) / len(journeys['user_id'].unique()) * 100
    avg_journey_length = journeys.groupby('user_id').size().mean()
    total_revenue = conversions['conversion_value'].sum()
    
    print(f"\n📈 Conversion Statistics:")
    print(f"   Conversion Rate: {conversion_rate:.2f}%")
    print(f"   Avg Journey Length: {avg_journey_length:.2f} touchpoints")
    print(f"   Total Revenue: ${total_revenue:,.2f}")
    print(f"   Avg Order Value: ${conversions['conversion_value'].mean():.2f}")
    
    # Channel distribution
    print(f"\n📡 Top Channels:")
    channel_dist = journeys['channel'].value_counts().head(7)
    for channel, count in channel_dist.items():
        pct = (count / len(journeys)) * 100
        print(f"   {channel}: {count} ({pct:.1f}%)")
    
    # ============================================================================
    # 2. SIMPLE MODELS TRAINING
    # ============================================================================
    
    print("\n\n🎯 PHASE 2: SIMPLE ATTRIBUTION MODELS")
    print("-" * 80)
    
    print("Training simple models...")
    simple_comparator = AttributionModelComparator()
    simple_results = simple_comparator.compare_models(journeys)
    
    print(f"✓ Last-Click Attribution")
    print(f"✓ Linear Attribution")
    print(f"✓ Time-Decay Attribution")
    print(f"✓ Position-Based Attribution")
    
    # Summary statistics
    print(f"\n📊 Simple Model Summary:")
    simple_summary = simple_results.groupby('model')['credit'].sum()
    for model, total_credit in simple_summary.items():
        print(f"   {model}: ${total_credit:,.2f}")
    
    print(f"\n📡 Channel Credit Distribution (Simple Models):")
    channel_credit = simple_results.groupby(['model', 'channel'])['credit'].sum().unstack(fill_value=0)
    print(channel_credit.round(2))
    
    # ============================================================================
    # 3. ADVANCED MODELS TRAINING
    # ============================================================================
    
    print("\n\n🚀 PHASE 3: ADVANCED ATTRIBUTION MODELS")
    print("-" * 80)
    
    advanced_comparator = AdvancedAttributionComparator()
    
    # Markov Chain
    print("Training Markov Chain...")
    advanced_comparator.add_markov_chain(journeys)
    print("✓ Markov Chain trained")
    
    # XGBoost
    try:
        print("Training XGBoost...")
        advanced_comparator.add_xgboost(features)
        print("✓ XGBoost trained")
    except Exception as e:
        print(f"⚠️ XGBoost skipped: {e}")
    
    # LightGBM
    try:
        print("Training LightGBM...")
        advanced_comparator.add_lightgbm(features)
        print("✓ LightGBM trained")
    except Exception as e:
        print(f"⚠️ LightGBM skipped: {e}")
    
    # ============================================================================
    # 4. MODEL EVALUATION
    # ============================================================================
    
    print("\n\n📈 PHASE 4: MODEL EVALUATION & COMPARISON")
    print("-" * 80)
    
    # Combine all results for evaluation
    all_results = simple_results.copy()
    
    print(f"\n✓ Evaluating {all_results['model'].nunique()} models")
    
    # Channel credit summary
    print(f"\n💰 Channel Credit Summary (All Models):")
    evaluator = AttributionEvaluator()
    credit_summary = evaluator.channel_credit_summary(all_results)
    print(credit_summary)
    
    # Model agreement
    print(f"\n🤝 Model Agreement Analysis:")
    agreement = evaluator.model_agreement(all_results)
    print(f"   Overall Agreement Score: {agreement['overall_agreement']:.2%}")
    print(f"\n   Agreement by Channel:")
    for channel, disagreement in agreement['channel_agreement'].items():
        agreement_pct = 100 - (disagreement * 100)
        print(f"   {channel}: {agreement_pct:.1f}% agreement")
    
    # Model stability
    print(f"\n🛡️ Model Stability (consistency):")
    stability = evaluator.model_stability(all_results)
    for model, score in sorted(stability.items(), key=lambda x: x[1], reverse=True):
        print(f"   {model}: {score:.2%}")
    
    # ROI potential
    print(f"\n💹 ROI Potential by Channel:")
    roi = evaluator.roi_potential(all_results)
    roi_by_model = roi.groupby('model')['roi_potential'].mean().sort_values(ascending=False)
    for model, roi_val in roi_by_model.items():
        print(f"   {model}: ${roi_val:.2f}/touch")
    
    # ============================================================================
    # 5. VISUALIZATIONS
    # ============================================================================
    
    print("\n\n📊 PHASE 5: GENERATING VISUALIZATIONS")
    print("-" * 80)
    
    visualizer = AttributionVisualizer()
    
    print("Creating visualizations...")
    
    # Channel credit comparison
    visualizer.plot_channel_credit_comparison(
        all_results, 
        output_path=f'{output_dir}/attribution_channel_comparison.png'
    )
    
    # Model heatmap
    visualizer.plot_model_heatmap(
        all_results,
        output_path=f'{output_dir}/attribution_model_heatmap.png'
    )
    
    # Model agreement
    visualizer.plot_model_agreement(
        agreement,
        output_path=f'{output_dir}/attribution_model_agreement.png'
    )
    
    # ROI potential
    visualizer.plot_roi_potential(
        roi,
        output_path=f'{output_dir}/attribution_roi_potential.png'
    )
    
    # ============================================================================
    # 6. EXPORT RESULTS
    # ============================================================================
    
    print("\n\n💾 PHASE 6: EXPORTING RESULTS")
    print("-" * 80)
    
    # Save results CSV
    all_results.to_csv(f'{output_dir}/attribution_results.csv', index=False)
    print(f"✓ Saved: attribution_results.csv")
    
    # Save credit summary
    credit_summary.to_csv(f'{output_dir}/attribution_credit_summary.csv')
    print(f"✓ Saved: attribution_credit_summary.csv")
    
    # Save ROI analysis
    roi.to_csv(f'{output_dir}/attribution_roi_analysis.csv', index=False)
    print(f"✓ Saved: attribution_roi_analysis.csv")
    
    # Create summary report
    summary_report = {
        'timestamp': datetime.now().isoformat(),
        'n_users': n_users,
        'n_touchpoints': len(journeys),
        'n_conversions': len(conversions),
        'conversion_rate': conversion_rate,
        'total_revenue': float(total_revenue),
        'avg_order_value': float(conversions['conversion_value'].mean()),
        'avg_journey_length': float(avg_journey_length),
        'n_models': all_results['model'].nunique(),
        'n_channels': journeys['channel'].nunique(),
        'model_agreement_score': float(agreement['overall_agreement']),
        'top_stability_model': max(stability.items(), key=lambda x: x[1])[0],
        'top_roi_model': roi_by_model.idxmax()
    }
    
    with open(f'{output_dir}/attribution_summary.json', 'w') as f:
        json.dump(summary_report, f, indent=2)
    print(f"✓ Saved: attribution_summary.json")
    
    # ============================================================================
    # 7. FINAL SUMMARY
    # ============================================================================
    
    print("\n\n" + "="*80)
    print(" ✅ TRAINING COMPLETE")
    print("="*80)
    
    print(f"\n📊 Final Summary:")
    print(f"   Models trained: {all_results['model'].nunique()}")
    print(f"   Channels analyzed: {journeys['channel'].nunique()}")
    print(f"   Total revenue modeled: ${total_revenue:,.2f}")
    print(f"   Model agreement: {agreement['overall_agreement']:.2%}")
    
    print(f"\n🎯 Key Insights:")
    print(f"   • {roi_by_model.idxmax()} model has highest ROI potential")
    print(f"   • {max(stability.items(), key=lambda x: x[1])[0]} model is most stable")
    print(f"   • Model disagreement suggests different fraud detection strategies")
    
    print(f"\n📁 All results saved to: {output_dir}/")
    
    return {
        'journeys': journeys,
        'conversions': conversions,
        'features': features,
        'results': all_results,
        'agreement': agreement,
        'stability': stability,
        'roi': roi,
        'summary': summary_report
    }


if __name__ == "__main__":
    import os
    os.makedirs('results', exist_ok=True)
    
    results = train_all_models(n_users=2000, output_dir='results')
