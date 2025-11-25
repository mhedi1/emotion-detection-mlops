"""
Model training script with real-time augmentation
Trains emotion detection model using transfer learning
"""

import os
import yaml
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import mlflow
import mlflow.keras
from pathlib import Path
import logging
from sklearn.utils.class_weight import compute_class_weight

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def load_params():
    """Load parameters from params.yaml"""
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params


def calculate_class_weights(train_generator):
    """Calculate class weights to handle imbalance"""
    class_counts = np.bincount(train_generator.classes)
    total_samples = len(train_generator.classes)
    num_classes = len(class_counts)
    
    # Calculate balanced class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.arange(num_classes),
        y=train_generator.classes
    )
    
    class_weights_dict = dict(enumerate(class_weights))
    
    logger.info("Class distribution and weights:")
    for class_idx, class_name in enumerate(train_generator.class_indices.keys()):
        logger.info(f"  {class_name}: {class_counts[class_idx]} samples, weight: {class_weights_dict[class_idx]:.2f}")
    
    return class_weights_dict


def create_data_generators(params):
    """
    Create training and validation data generators with real-time augmentation
    
    Args:
        params: Configuration parameters
    
    Returns:
        train_generator, val_generator, test_generator
    """
    image_size = tuple(params['data']['image_size'])
    batch_size = params['training']['batch_size']
    
    # Choose data source: augmented if available, otherwise processed
    if params['augmentation']['use_for_training'] and Path("data/processed/train").exists():
        data_dir = "data/processed"
        logger.info("Using processed dataset for training")
    else:
        data_dir = "data/processed" 
        logger.info("Using processed dataset for training")
    
    # Training data generator with augmentation
    aug_params = params['preprocessing']['augmentation']
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=aug_params['rotation_range'],
        horizontal_flip=aug_params['horizontal_flip'],
        width_shift_range=aug_params['width_shift_range'],
        height_shift_range=aug_params['height_shift_range'],
        brightness_range=aug_params['brightness_range'],
        zoom_range=aug_params['zoom_range'],
        fill_mode=aug_params['fill_mode']
    )
    
    # Validation and test generators WITHOUT augmentation
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        f'{data_dir}/train',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        f'{data_dir}/val',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        f'{data_dir}/test',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    logger.info("✓ Data generators created:")
    logger.info(f"  Training samples: {train_generator.samples}")
    logger.info(f"  Validation samples: {val_generator.samples}")
    logger.info(f"  Test samples: {test_generator.samples}")
    logger.info(f"  Classes: {list(train_generator.class_indices.keys())}")
    
    return train_generator, val_generator, test_generator


def build_model(params, num_classes):
    """
    Build model using transfer learning with MobileNetV2
    
    Args:
        params: Configuration parameters
        num_classes: Number of emotion classes
    
    Returns:
        Compiled Keras model
    """
    image_size = tuple(params['data']['image_size'])
    model_config = params['model']
    
    # Load pre-trained base model
    if model_config['architecture'] == "MobileNetV2":
        base_model = keras.applications.MobileNetV2(
            input_shape=(*image_size, 3),
            include_top=False,
            weights='imagenet'
        )
    elif model_config['architecture'] == "EfficientNetB0":
        base_model = keras.applications.EfficientNetB0(
            input_shape=(*image_size, 3),
            include_top=False,
            weights='imagenet'
        )
    else:
        raise ValueError(f"Unsupported architecture: {model_config['architecture']}")
    
    # Configure trainable layers based on params
    if model_config['freeze_base']:
        base_model.trainable = False
        logger.info("✓ Base model frozen (feature extraction)")
    else:
        base_model.trainable = True
        logger.info("✓ Base model unfrozen (fine-tuning)")
    
    # Build custom classifier on top
    inputs = keras.Input(shape=(*image_size, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(model_config['dropout_rate'])(x)
    
    # Add dense layers with regularization
    x = layers.Dense(model_config['dense_units'], 
                    activation=model_config['activation'],
                    kernel_regularizer=keras.regularizers.l2(model_config.get('l2_regularization', 0.01)))(x)
    
    if model_config.get('use_batch_norm', True):
        x = layers.BatchNormalization()(x)
    
    x = layers.Dropout(model_config['dropout_rate'])(x)
    outputs = layers.Dense(num_classes, activation=model_config['final_activation'])(x)
    
    model = keras.Model(inputs, outputs)
    
    # Compile model - FIXED: Ensure learning rate is float
    initial_learning_rate = float(params['training']['initial_learning_rate'])
    
    model.compile(
        optimizer=keras.optimizers.Adam(initial_learning_rate),
        loss=params['training']['loss'],
        metrics=['accuracy', 
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )
    
    logger.info("✓ Model built and compiled")
    logger.info(f"  Total parameters: {model.count_params():,}")
    trainable_count = sum([tf.size(w).numpy() for w in model.trainable_weights])
    logger.info(f"  Trainable parameters: {trainable_count:,}")
    
    return model


def create_callbacks(params):
    """
    Create training callbacks
    
    Args:
        params: Configuration parameters
    
    Returns:
        List of callbacks
    """
    callbacks = []
    
    # Model checkpoint
    if params['training']['checkpointing']['enabled']:
        checkpoint_dir = params['paths']['checkpoints']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(checkpoint_dir, 'best_model.h5'),
            monitor=params['training']['checkpointing']['monitor'],
            save_best_only=True,
            mode='max',
            verbose=1
        )
        callbacks.append(checkpoint)
        logger.info("✓ Model checkpoint enabled")
    
    # Early stopping
    if params['training']['early_stopping']['enabled']:
        early_stop = keras.callbacks.EarlyStopping(
            monitor=params['training']['early_stopping']['monitor'],
            patience=params['training']['early_stopping']['patience'],
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stop)
        logger.info(f"✓ Early stopping enabled (patience={params['training']['early_stopping']['patience']})")
    
    # Learning rate reduction - FIXED: Ensure all values are floats
    lr_config = params['training']['learning_rate_schedule']
    if lr_config['enabled']:
        # Convert all values to float to avoid string comparison issues
        reduce_lr_factor = float(lr_config['reduce_lr_factor'])
        reduce_lr_patience = int(lr_config['reduce_lr_patience'])
        reduce_lr_min = float(lr_config['reduce_lr_min'])
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_lr=reduce_lr_min,
            verbose=1
        )
        callbacks.append(reduce_lr)
        logger.info("✓ Learning rate scheduling enabled")
    
    # TensorBoard (optional)
    if params['monitoring']['use_tensorboard']:
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=params['paths']['tensorboard_logs'],
            update_freq=params['monitoring']['tensorboard_update_freq']
        )
        callbacks.append(tensorboard_callback)
        logger.info("✓ TensorBoard logging enabled")
    
    return callbacks


def main():
    """Main training pipeline"""
    logger.info("=" * 60)
    logger.info("EMOTION DETECTION MODEL TRAINING")
    logger.info("=" * 60)
    
    # Load parameters
    params = load_params()
    logger.info("✓ Parameters loaded")
    
    # Create data generators
    logger.info("\n" + "=" * 60)
    logger.info("Creating data generators...")
    logger.info("=" * 60)
    train_gen, val_gen, test_gen = create_data_generators(params)
    num_classes = len(train_gen.class_indices)
    
    # Calculate class weights for handling imbalance
    logger.info("\n" + "=" * 60)
    logger.info("Calculating class weights...")
    logger.info("=" * 60)
    class_weights = calculate_class_weights(train_gen)
    
    # Build model
    logger.info("\n" + "=" * 60)
    logger.info("Building model...")
    logger.info("=" * 60)
    model = build_model(params, num_classes)
    
    # Create callbacks
    logger.info("\n" + "=" * 60)
    logger.info("Setting up training callbacks...")
    logger.info("=" * 60)
    callbacks = create_callbacks(params)
    
    # Setup MLflow
    logger.info("\n" + "=" * 60)
    logger.info("Setting up MLflow experiment tracking...")
    logger.info("=" * 60)
    
    mlflow.set_experiment(params['tracking']['experiment_name'])
    
    # Start MLflow run
    with mlflow.start_run(run_name=params['tracking']['run_name']):
        # Log parameters - FIXED: Ensure learning rate is float for logging
        initial_learning_rate = float(params['training']['initial_learning_rate'])
        
        mlflow.log_params({
            'model_architecture': params['model']['architecture'],
            'image_size': params['data']['image_size'],
            'batch_size': params['training']['batch_size'],
            'epochs': params['training']['epochs'],
            'learning_rate': initial_learning_rate,
            'num_classes': num_classes,
            'train_samples': train_gen.samples,
            'val_samples': val_gen.samples,
            'class_weights_used': True,
            'augmentation': params['preprocessing']['augmentation']['enabled']
        })
        
        # Log augmentation parameters
        for key, value in params['preprocessing']['augmentation'].items():
            mlflow.log_param(f'aug_{key}', value)
        
        logger.info("✓ MLflow tracking started")
        
        # Train model
        logger.info("\n" + "=" * 60)
        logger.info("STARTING TRAINING...")
        logger.info("=" * 60)
        
        history = model.fit(
            train_gen,
            epochs=params['training']['epochs'],
            validation_data=val_gen,
            callbacks=callbacks,
            class_weight=class_weights,  # Critical for handling imbalance
            verbose=1
        )
        
        # Evaluate on test set
        logger.info("\n" + "=" * 60)
        logger.info("Evaluating on test set...")
        logger.info("=" * 60)
        
        test_results = model.evaluate(test_gen, verbose=1)
        
        test_metrics = dict(zip(model.metrics_names, test_results))
        logger.info("\n✓ Test Results:")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Log final metrics to MLflow
        mlflow.log_metrics({
            'final_train_accuracy': history.history['accuracy'][-1],
            'final_val_accuracy': history.history['val_accuracy'][-1],
            'test_loss': test_metrics['loss'],
            'test_accuracy': test_metrics['accuracy'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall']
        })
        
        # Calculate F1 score
        if test_metrics['precision'] + test_metrics['recall'] > 0:
            test_f1 = 2 * (test_metrics['precision'] * test_metrics['recall']) / (test_metrics['precision'] + test_metrics['recall'])
            mlflow.log_metric('test_f1_score', test_f1)
            logger.info(f"  F1 Score: {test_f1:.4f}")
        
        # Save final model
        model_dir = params['paths']['saved_models']
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'emotion_model_final.h5')
        model.save(model_path)
        logger.info(f"\n✓ Model saved: {model_path}")
        
        # Log model to MLflow
        mlflow.keras.log_model(model, "model")
        
        # Log training history plots (optional)
        if params['monitoring']['save_training_plots']:
            try:
                import matplotlib.pyplot as plt
                
                # Plot training history
                plt.figure(figsize=(12, 4))
                
                plt.subplot(1, 2, 1)
                plt.plot(history.history['accuracy'], label='Training Accuracy')
                plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
                plt.title('Model Accuracy')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()
                
                plt.subplot(1, 2, 2)
                plt.plot(history.history['loss'], label='Training Loss')
                plt.plot(history.history['val_loss'], label='Validation Loss')
                plt.title('Model Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                
                plt.tight_layout()
                plt.savefig('training_history.png')
                mlflow.log_artifact('training_history.png')
                logger.info("✓ Training history plot saved")
            except Exception as e:
                logger.warning(f"Could not save training plots: {e}")
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ TRAINING COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"\n📊 View results: mlflow ui")
        logger.info(f"   Then open: http://localhost:5000")
        logger.info(f"\n💾 Model saved to: {model_path}")
        
        # Final performance summary
        logger.info(f"\n🎯 FINAL PERFORMANCE:")
        logger.info(f"   Test Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"   Test Precision: {test_metrics['precision']:.4f}")
        logger.info(f"   Test Recall: {test_metrics['recall']:.4f}")


if __name__ == "__main__":
    main()