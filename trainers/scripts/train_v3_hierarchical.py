import os
import sys
import json
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trainers.hierarchical.hierarchical_trainer import HierarchicalModelTrainer

# Configuration
DATA_ROOT = "/Users/Williamhiler/Documents/my-project/train/train-data"
MODEL_DIR = "/Users/Williamhiler/Documents/my-project/models"
SEASONS = [f"{year}-{year+1}" for year in range(2015, 2025)]  # 2015-2016 to 2024-2025 seasons


def train_hierarchical_all_data():
    """Train hierarchical model using all seasons combined"""
    trainer = HierarchicalModelTrainer(DATA_ROOT, MODEL_DIR)
    results = {}
    
    # Train for all seasons combined
    print(f"\n{'='*60}")
    print(f"Training for all seasons combined (2015-2024)")
    print(f"{'='*60}")
    metrics = trainer.train(SEASONS)
    results["all_seasons"] = metrics
    
    # Save results
    results_file = os.path.join(MODEL_DIR, "v3", "3.0.7", "hierarchical_model_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"All training completed!")
    print(f"Results saved to: {results_file}")
    
    # Print summary
    print(f"{'='*60}")
    print(f"Performance Summary for Hierarchical Model v3.0.7 (LightGBM)")
    print(f"{'='*60}")
    for key, model_info in results.items():
        print(f"\n{key}:")
        print(f"  Accuracy: {model_info['metrics']['accuracy']:.4f}")
        print(f"  Home Win F1: {model_info['metrics']['home_win_f1']:.4f}")
        print(f"  Draw F1: {model_info['metrics']['draw_f1']:.4f}")
        print(f"  Away Win F1: {model_info['metrics']['away_win_f1']:.4f}")
        print(f"  Weighted F1: {model_info['metrics']['weighted_f1']:.4f}")
    
    return results


if __name__ == "__main__":
    # Train with all data combined
    train_hierarchical_all_data()
