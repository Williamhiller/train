import os
import json
from trainers.hierarchical.hierarchical_trainer import HierarchicalModelTrainer

# Configuration
DATA_ROOT = "/Users/Williamhiler/Documents/my-project/train/train-data"
MODEL_DIR = "/Users/Williamhiler/Documents/my-project/models"
# Use all seasons from 2015-2016 to 2024-2025
SEASONS = [f"{year}-{year+1}" for year in range(2015, 2025)]  # 2015-2016 to 2024-2025 seasons

def train_hierarchical_all_data():
    """Train hierarchical model using all data combined"""
    print(f"{'='*60}")
    print(f"Training Hierarchical Model v3.0.8 (LightGBM) on ALL seasons combined")
    print(f"{'='*60}")
    print(f"Seasons: {SEASONS}")
    print(f"{'='*60}")
    
    trainer = HierarchicalModelTrainer(DATA_ROOT, MODEL_DIR)
    
    # Train with all seasons combined, enabling hyperparameter tuning
    metrics = trainer.train(SEASONS, use_tuning=True)
    
    print(f"{'='*60}")
    print(f"Training completed successfully!")
    print(f"{'='*60}")
    
    # Print detailed performance
    print(f"\n{'='*60}")
    print(f"FINAL MODEL PERFORMANCE")
    print(f"{'='*60}")
    print(f"Accuracy: {metrics['metrics']['accuracy']:.4f}")
    print(f"Home Win F1: {metrics['metrics']['home_win_f1']:.4f}")
    print(f"Draw F1: {metrics['metrics']['draw_f1']:.4f}")
    print(f"Away Win F1: {metrics['metrics']['away_win_f1']:.4f}")
    print(f"Weighted F1: {metrics['metrics']['weighted_f1']:.4f}")
    print(f"{'='*60}")
    
    print(f"\nModel saved to: {metrics['draw_model_path'].rsplit('/', 1)[0]}")
    print(f"Model Info: {metrics['draw_model_path'].replace('draw_model_all_seasons.joblib', 'model_info_all_seasons.json')}")
    print(f"Predictions: {metrics['draw_model_path'].replace('draw_model_all_seasons.joblib', 'predictions_all_seasons.csv')}")
    
    return metrics

if __name__ == "__main__":
    train_hierarchical_all_data()