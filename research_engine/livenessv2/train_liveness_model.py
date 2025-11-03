#!/usr/bin/env python3
"""
Professional Liveness Detection Training System
===============================================

This module provides a complete training pipeline for face liveness detection
with data augmentation, model training, evaluation, and export capabilities.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization, 
    Dropout, Flatten, Dense, Activation
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import json
import time
from pathlib import Path

def convert_numpy_types(obj):
    """
    Convert numpy types to native Python types for JSON serialization
    """
    if hasattr(obj, 'item'):
        return obj.item()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

class LivenessTrainer:
    """
    Professional training system for liveness detection models
    """
    
    def __init__(self, config=None):
        """
        Initialize the trainer
        
        Args:
            config (dict): Training configuration
        """
        self.config = config or self.get_default_config()
        self.model = None
        self.history = None
        
        # Create output directories
        self.create_directories()
        
        print("Liveness Detection Trainer initialized")
        print(f"Image size: {self.config['image_size']}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Epochs: {self.config['epochs']}")
    
    def get_default_config(self):
        """Get default training configuration"""
        return {
            'image_size': (128, 128),
            'batch_size': 32,
            'epochs': 50,
            'learning_rate': 0.001,
            'validation_split': 0.2,
            'data_augmentation': True,
            'early_stopping_patience': 10,
            'reduce_lr_patience': 5,
            'model_save_path': 'models',
            'logs_path': 'logs',
            'results_path': 'results'
        }
    
    def create_directories(self):
        """Create necessary directories"""
        for path in ['model_save_path', 'logs_path', 'results_path']:
            Path(self.config[path]).mkdir(parents=True, exist_ok=True)
    
    def build_model(self, input_shape=None, num_classes=2):
        """
        Build the CNN model for liveness detection
        
        Args:
            input_shape (tuple): Input image shape
            num_classes (int): Number of output classes
            
        Returns:
            tensorflow.keras.Model: Compiled model
        """
        if input_shape is None:
            input_shape = (*self.config['image_size'], 3)
        
        model = Sequential([
            # First Convolutional Block
            Conv2D(32, (3, 3), activation='relu', padding='same', 
                   input_shape=input_shape, name='conv1'),
            BatchNormalization(name='bn1'),
            Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2'),
            BatchNormalization(name='bn2'),
            MaxPooling2D((2, 2), name='pool1'),
            Dropout(0.25, name='dropout1'),
            
            # Second Convolutional Block
            Conv2D(64, (3, 3), activation='relu', padding='same', name='conv3'),
            BatchNormalization(name='bn3'),
            Conv2D(64, (3, 3), activation='relu', padding='same', name='conv4'),
            BatchNormalization(name='bn4'),
            MaxPooling2D((2, 2), name='pool2'),
            Dropout(0.25, name='dropout2'),
            
            # Third Convolutional Block
            Conv2D(128, (3, 3), activation='relu', padding='same', name='conv5'),
            BatchNormalization(name='bn5'),
            Conv2D(128, (3, 3), activation='relu', padding='same', name='conv6'),
            BatchNormalization(name='bn6'),
            MaxPooling2D((2, 2), name='pool3'),
            Dropout(0.25, name='dropout3'),
            
            # Fully Connected Layers
            Flatten(name='flatten'),
            Dense(512, activation='relu', name='fc1'),
            BatchNormalization(name='bn_fc1'),
            Dropout(0.5, name='dropout_fc1'),
            
            Dense(256, activation='relu', name='fc2'),
            BatchNormalization(name='bn_fc2'),
            Dropout(0.5, name='dropout_fc2'),
            
            # Output Layer
            Dense(num_classes, activation='softmax', name='predictions')
        ])
        
        # Compile model
        optimizer = Adam(learning_rate=self.config['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def prepare_data_generators(self, data_dir, validation_split=0.2):
        """
        Prepare data generators for training
        
        Args:
            data_dir (str): Path to data directory
            validation_split (float): Validation split ratio
            
        Returns:
            tuple: (train_generator, validation_generator)
        """
        # Data augmentation for training
        if self.config['data_augmentation']:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                brightness_range=[0.8, 1.2],
                validation_split=validation_split
            )
        else:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=validation_split
            )
        
        # Validation data (no augmentation)
        validation_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.config['image_size'],
            batch_size=self.config['batch_size'],
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        validation_generator = validation_datagen.flow_from_directory(
            data_dir,
            target_size=self.config['image_size'],
            batch_size=self.config['batch_size'],
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        return train_generator, validation_generator
    
    def prepare_callbacks(self):
        """
        Prepare training callbacks
        
        Returns:
            list: List of callbacks
        """
        callbacks = []
        
        # Model checkpoint
        checkpoint_path = os.path.join(
            self.config['model_save_path'], 
            'best_model.h5'
        )
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.config['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Reduce learning rate
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=self.config['reduce_lr_patience'],
            min_lr=0.0001,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        return callbacks
    
    def train(self, data_dir):
        """
        Train the liveness detection model
        
        Args:
            data_dir (str): Path to training data directory
        """
        print("Starting training...")
        print(f"Data directory: {data_dir}")
        
        # Check if data directory exists
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory not found: {data_dir}")
        
        # Build model
        if self.model is None:
            self.build_model()
        
        print(f"Model architecture:")
        self.model.summary()
        
        # Prepare data
        train_gen, val_gen = self.prepare_data_generators(
            data_dir, self.config['validation_split']
        )
        
        print(f"Training samples: {train_gen.samples}")
        print(f"Validation samples: {val_gen.samples}")
        print(f"Classes: {train_gen.class_indices}")
        
        # Prepare callbacks
        callbacks = self.prepare_callbacks()
        
        # Train model
        start_time = time.time()
        
        self.history = self.model.fit(
            train_gen,
            epochs=self.config['epochs'],
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Save final model
        final_model_path = os.path.join(
            self.config['model_save_path'], 
            'final_model.h5'
        )
        self.model.save(final_model_path)
        print(f"Final model saved to: {final_model_path}")
        
        # Save training history
        self.save_training_history()
        
        # Generate training report
        self.generate_training_report(train_gen, val_gen, training_time)
    
    def save_training_history(self):
        """Save training history to JSON file"""
        if self.history is None:
            return
        
        history_path = os.path.join(self.config['logs_path'], 'training_history.json')
        
        # Convert numpy arrays to lists for JSON serialization
        history_dict = {}
        for key, values in self.history.history.items():
            history_dict[key] = [float(v) for v in values]
        
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        print(f"Training history saved to: {history_path}")
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history
        
        Args:
            save_path (str): Path to save plots
        """
        if self.history is None:
            print("No training history available")
            return
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training plots saved to: {save_path}")
        
        plt.show()
    
    def evaluate_model(self, test_dir):
        """
        Evaluate trained model on test data
        
        Args:
            test_dir (str): Path to test data directory
        """
        if self.model is None:
            print("No model to evaluate")
            return
        
        print("Evaluating model...")
        
        # Prepare test data generator
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=self.config['image_size'],
            batch_size=self.config['batch_size'],
            class_mode='categorical',
            shuffle=False
        )
        
        # Evaluate
        test_loss, test_accuracy = self.model.evaluate(test_generator, verbose=1)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Get predictions
        predictions = self.model.predict(test_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_generator.classes
        
        # Generate classification report
        class_names = list(test_generator.class_indices.keys())
        report = classification_report(
            true_classes, predicted_classes, 
            target_names=class_names, 
            output_dict=True
        )
        
        print("\nClassification Report:")
        print(classification_report(true_classes, predicted_classes, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Save evaluation results
        results = {
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss),
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
        
        # Convert numpy types to native Python types
        results = convert_numpy_types(results)
        
        results_path = os.path.join(self.config['results_path'], 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Evaluation results saved to: {results_path}")
        
        return results
    
    def generate_training_report(self, train_gen, val_gen, training_time):
        """
        Generate comprehensive training report
        
        Args:
            train_gen: Training data generator
            val_gen: Validation data generator
            training_time (float): Training time in seconds
        """
        report = {
            'training_config': self.config,
            'model_summary': {
                'total_params': self.model.count_params(),
                'trainable_params': sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights]),
                'non_trainable_params': sum([tf.keras.backend.count_params(w) for w in self.model.non_trainable_weights])
            },
            'data_info': {
                'training_samples': train_gen.samples,
                'validation_samples': val_gen.samples,
                'classes': train_gen.class_indices,
                'image_size': self.config['image_size'],
                'batch_size': self.config['batch_size']
            },
            'training_results': {
                'epochs_completed': len(self.history.history['loss']),
                'training_time_seconds': training_time,
                'final_training_accuracy': float(self.history.history['accuracy'][-1]),
                'final_validation_accuracy': float(self.history.history['val_accuracy'][-1]),
                'final_training_loss': float(self.history.history['loss'][-1]),
                'final_validation_loss': float(self.history.history['val_loss'][-1]),
                'best_validation_accuracy': float(max(self.history.history['val_accuracy']))
            }
        }
        
        # Convert numpy types to native Python types
        report = convert_numpy_types(report)
        
        # Save report
        report_path = os.path.join(self.config['results_path'], 'training_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Training report saved to: {report_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Training Time: {training_time:.2f} seconds")
        print(f"Epochs Completed: {report['training_results']['epochs_completed']}")
        print(f"Final Training Accuracy: {report['training_results']['final_training_accuracy']:.4f}")
        print(f"Final Validation Accuracy: {report['training_results']['final_validation_accuracy']:.4f}")
        print(f"Best Validation Accuracy: {report['training_results']['best_validation_accuracy']:.4f}")
        print(f"Model Parameters: {report['model_summary']['total_params']:,}")
        print("="*60)

def create_sample_data_structure():
    """
    Create sample data directory structure for training
    """
    print("Creating sample data structure...")
    
    # Create directories
    base_dir = "liveness_data"
    subdirs = ["train/real", "train/fake", "validation/real", "validation/fake", "test/real", "test/fake"]
    
    for subdir in subdirs:
        Path(f"{base_dir}/{subdir}").mkdir(parents=True, exist_ok=True)
    
    # Create README with instructions
    readme_content = """
# Liveness Detection Training Data

## Directory Structure
```
liveness_data/
├── train/
│   ├── real/          # Real face images for training
│   └── fake/          # Fake face images for training
├── validation/
│   ├── real/          # Real face images for validation
│   └── fake/          # Fake face images for validation
└── test/
    ├── real/          # Real face images for testing
    └── fake/          # Fake face images for testing
```

## Data Collection Guidelines

### Real Face Images:
- Capture from real people using webcam
- Various lighting conditions
- Different poses and expressions
- Multiple people of different ages/ethnicities
- At least 1000+ images recommended

### Fake Face Images:
- Photos displayed on phone screens
- Printed photos
- Photos displayed on computer monitors
- Video playback attacks
- At least 1000+ images recommended

## Image Requirements:
- Face should be clearly visible
- Minimum size: 128x128 pixels
- Format: JPG, PNG
- Good quality, not blurry

## Usage:
1. Collect and organize your data in the above structure
2. Run the training script
3. The model will be automatically trained and saved
"""
    
    with open(f"{base_dir}/README.md", 'w') as f:
        f.write(readme_content)
    
    print(f"Sample data structure created in: {base_dir}/")
    print("Please collect your training data according to the README instructions")

def main():
    """Main training function"""
    print("=" * 70)
    print("🚀 PROFESSIONAL LIVENESS DETECTION TRAINING SYSTEM 🚀")
    print("=" * 70)
    
    # Configuration
    config = {
        'image_size': (128, 128),
        'batch_size': 32,
        'epochs': 50,
        'learning_rate': 0.001,
        'validation_split': 0.2,
        'data_augmentation': True,
        'early_stopping_patience': 10,
        'reduce_lr_patience': 5,
        'model_save_path': 'models',
        'logs_path': 'logs',
        'results_path': 'results'
    }
    
    # Check if data exists
    data_dir = "liveness_data/train"
    if not os.path.exists(data_dir):
        print("Training data not found!")
        print("Creating sample data structure...")
        create_sample_data_structure()
        print("\nPlease collect your training data first, then run this script again.")
        return
    
    # Initialize trainer
    trainer = LivenessTrainer(config)
    
    try:
        # Train model
        trainer.train(data_dir)
        
        # Plot training history
        plot_path = os.path.join(config['results_path'], 'training_plots.png')
        trainer.plot_training_history(plot_path)
        
        # Evaluate on test data if available
        test_dir = "liveness_data/test"
        if os.path.exists(test_dir):
            trainer.evaluate_model(test_dir)
        else:
            print("Test data not found, skipping evaluation")
        
        print("\n🎉 Training completed successfully!")
        print(f"📁 Models saved in: {config['model_save_path']}/")
        print(f"📊 Results saved in: {config['results_path']}/")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()