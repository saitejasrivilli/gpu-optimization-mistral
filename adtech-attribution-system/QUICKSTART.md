#!/usr/bin/env python3
"""
AdTech Attribution System - Quick Start
Run this to generate complete attribution analysis
"""

import os
import sys

def run_demo():
    """Run full attribution pipeline demo"""
    
    print("\n" + "="*80)
    print(" ADTECH ATTRIBUTION SYSTEM - QUICK START")
    print("="*80 + "\n")
    
    # Check dependencies
    print("📦 Checking dependencies...")
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import xgboost
        print("✓ All dependencies installed\n")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Run: pip install -r requirements.txt\n")
        return False
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Run training pipeline
    print("🚀 Running attribution training pipeline...")
    print("-" * 80)
    
    try:
        from main import train_all_models
        
        # Train with 2000 users (smaller for demo, use 10000+ for full analysis)
        results = train_all_models(n_users=2000, output_dir='results')
        
        print("\n" + "="*80)
        print(" ✅ COMPLETE!")
        print("="*80 + "\n")
        
        print("📁 Generated Files:")
        print("   results/attribution_results.csv          - Full attribution data")
        print("   results/attribution_summary.json         - Summary metrics")
        print("   results/attribution_channel_comparison.png - Channel credit heatmap")
        print("   results/attribution_model_heatmap.png    - Model comparison")
        print("   results/attribution_roi_potential.png    - ROI analysis")
        
        print("\n📊 Key Insights:")
        summary = results['summary']
        print(f"   Conversion Rate: {summary['conversion_rate']:.2f}%")
        print(f"   Total Revenue Modeled: ${summary['total_revenue']:,.0f}")
        print(f"   Average Journey Length: {summary['avg_journey_length']:.2f} touches")
        print(f"   Model Agreement: {summary['model_agreement_score']:.2%}")
        print(f"   Top Stability Model: {summary['top_stability_model']}")
        print(f"   Top ROI Model: {summary['top_roi_model']}")
        
        print("\n🎯 Next Steps:")
        print("   1. Review the visualizations in results/")
        print("   2. Study the attribution_results.csv for detailed data")
        print("   3. Compare different models and understand their differences")
        print("   4. Implement the production API (Phase 2)")
        
        print("\n💡 Key Takeaways:")
        print("   • Last-Click is simplest but biased")
        print("   • Linear/Position-Based are fair compromises")
        print("   • Markov Chain is theoretically sound but complex")
        print("   • XGBoost learns from data but needs lots of it")
        print("   • Different models → different budget allocations")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_demo()
    sys.exit(0 if success else 1)
