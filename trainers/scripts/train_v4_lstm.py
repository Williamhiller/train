import os
import json
from trainers.neural_network.lstm_trainer import LSTMTrainer

# Configuration
DATA_ROOT = "/Users/Williamhiler/Documents/my-project/train/train-data"
MODEL_DIR = "/Users/Williamhiler/Documents/my-project/models"
# Use all seasons from 2015-2016 to 2024-2025
SEASONS = [f"{year}-{year+1}" for year in range(2015, 2025)]  # 2015-2016 to 2024-2025 seasons

def train_lstm_all_data():
    """Train LSTM model using all seasons combined"""
    print(f"{'='*60}")
    print(f"Training LSTM Model v4.0.0 on ALL seasons combined")
    print(f"{'='*60}")
    print(f"Seasons: {SEASONS}")
    print(f"{'='*60}")
    
    trainer = LSTMTrainer(DATA_ROOT, MODEL_DIR)
    
    # Train with all seasons combined
    metrics = trainer.train(SEASONS, use_tuning=False)
    
    print(f"{'='*60}")
    print(f"Training completed successfully!")
    print(f"{'='*60}")
    
    # Print detailed performance
    print(f"\n{'='*60}")
    print(f"FINAL LSTM MODEL PERFORMANCE")
    print(f"{'='*60}")
    print(f"Accuracy: {metrics['metrics']['accuracy']:.4f}")
    print(f"Home Win F1: {metrics['metrics']['home_win_f1']:.4f}")
    print(f"Draw F1: {metrics['metrics']['draw_f1']:.4f}")
    print(f"Away Win F1: {metrics['metrics']['away_win_f1']:.4f}")
    print(f"Weighted F1: {metrics['metrics']['weighted_f1']:.4f}")
    print(f"{'='*60}")
    
    print(f"\nModel saved to: {metrics['model_path']}")
    print(f"Model Info: {metrics['model_path'].replace('lstm_model_4.0.0.h5', 'model_info_4.0.0.json')}")
    return metrics

if __name__ == "__main__":
    train_lstm_all_data()