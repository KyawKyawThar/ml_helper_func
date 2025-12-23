# ============================================================================
# IMPORTS
# ============================================================================

import os
import json
import time
import zipfile
from datetime import datetime
from typing import Tuple, Dict, Optional, List, Callable, Union

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers, Sequential, Model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import (
    confusion_matrix, accuracy_score, 
    precision_recall_fscore_support, classification_report
)
import itertools
import random


# ============================================================================
# SECTION 1: DATA PROCESSING & PREPARATION
# ============================================================================

class ImageProcessor:
    """Utility class for image loading and preprocessing."""
    
    @staticmethod
    def load_and_prep_image(filename: str, image_shape: int = 224) -> tf.Tensor:
        """
        Read an image from filename and prepare it for model input.
        
        Args:
            filename: Path to the image file
            image_shape: Target image size (default 224x224)
            
        Returns:
            Preprocessed image tensor
        """
        img = tf.io.read_file(filename)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.resize(img, size=[image_shape, image_shape])
        img = img / 255
        return img
    
    @staticmethod
    def prepare_image(image: tf.Tensor, image_shape: int = 224, scale: bool = True) -> tf.Tensor:
        """
        Prepare an image tensor for model inference.
        
        Args:
            image: Input image tensor
            image_shape: Target image size (default 224)
            scale: Whether to scale to [0, 1] range (default True)
            
        Returns:
            Prepared image tensor
        """
        img = tf.image.resize(image, [image_shape, image_shape])
        if scale:
            return img / 255.
        else:
            return img


def walk_through_directory(dir_path: str) -> None:
    """
    Display directory structure and file counts.
    
    Args:
        dir_path: Path to the directory to analyze
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


def view_random_image(target_dir: str, target_class: str) -> np.ndarray:
    """
    Display a random image from the specified class directory.
    
    Args:
        target_dir: Path to the target directory
        target_class: Name of the class directory
        
    Returns:
        The loaded image array
    """
    target_folder = target_dir + target_class
    random_image = random.sample(os.listdir(target_folder), 1)
    print(f"Displaying random image from {target_folder}: {random_image}")
    img = mpimg.imread(target_folder + '/' + random_image[0])
    plt.imshow(img)
    plt.axis('off')
    plt.title(target_class)
    return img


def unzip_data(filename: str) -> None:
    """
    Unzip a file to the current working directory.
    
    Args:
        filename: Path to the zip file
    """
    zip_ref = zipfile.ZipFile(filename, "r")
    zip_ref.extractall()
    zip_ref.close()


# ============================================================================
# SECTION 2: DATA AUGMENTATION
# ============================================================================

def create_data_augmentation() -> Sequential:
    """
    Create a data augmentation pipeline.
    
    Returns:
        Sequential model with augmentation layers
    """
    return Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomHeight(0.2),
        layers.RandomWidth(0.2)
    ], name="data_augmentation")


# ============================================================================
# SECTION 3: MODEL CREATION & CONFIGURATION
# ============================================================================

def create_model(base_model: Model, image_shape: Tuple[int, int], 
                 num_classes: int = 10) -> Model:
    """
    Create a transfer learning model with dynamic preprocessing.
    
    Handles different normalization requirements for different base models:
    - EfficientNet: [0, 255] (no external scaling)
    - ResNetV2/MobileNet: [-1, 1] scaling
    - VGG16/DenseNet: [0, 1] scaling
    
    Args:
        base_model: Pre-trained base model
        image_shape: Input image shape (height, width)
        num_classes: Number of output classes (default 10)
        
    Returns:
        Compiled transfer learning model
    """
    data_augmentation = create_data_augmentation()
    
    # Input layer
    inputs = tf.keras.Input(shape=image_shape + (3,))
    
    # Augmentation
    x = data_augmentation(inputs)
    
    # Dynamic preprocessing based on model type
    model_name = base_model.name.lower()
    
    if "efficientnet" in model_name:
        print(f"Model: {base_model.name} -> Skipping external Rescaling (uses [0, 255]).")
        pass
    elif "resnet" in model_name and "v2" in model_name:
        print(f"Model: {base_model.name} -> Adding Rescaling [-1, 1].")
        x = layers.Rescaling(1./127.5, offset=-1)(x)
    elif "mobilenet" in model_name:
        print(f"Model: {base_model.name} -> Adding Rescaling [-1, 1].")
        x = layers.Rescaling(1./127.5, offset=-1)(x)
    else:
        print(f"Model: {base_model.name} -> Defaulting to Rescaling [0, 1].")
        x = layers.Rescaling(1./255.0)(x)
    
    # Feature extraction
    x = base_model(x, training=False)
    
    # Pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Classification head
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)


# ============================================================================
# SECTION 4: TRAINING & CALLBACKS
# ============================================================================

class TrainingMetrics:
    """Track and log training metrics across multiple phases."""
    
    def __init__(self, log_dir: str = "training_metrics"):
        """
        Initialize metrics tracker.
        
        Args:
            log_dir: Directory to save metrics logs
        """
        self.log_dir = log_dir
        self.metrics = {
            "epochs": [],
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "learning_rate": [],
            "training_time": []
        }
        os.makedirs(log_dir, exist_ok=True)
    
    def record(self, history_dict: Dict) -> None:
        """
        Record metrics from training history.
        
        Args:
            history_dict: Dictionary or History object from model.fit()
        """
        history = history_dict.history if hasattr(history_dict, 'history') else history_dict
        
        self.metrics["train_loss"].extend(history.get("loss", []))
        self.metrics["train_accuracy"].extend(history.get("accuracy", []))
        self.metrics["val_loss"].extend(history.get("val_loss", []))
        self.metrics["val_accuracy"].extend(history.get("val_accuracy", []))
        self.metrics["epochs"].extend(range(len(history.get("loss", []))))
    
    def save(self, filename: str = "metrics.json") -> None:
        """Save metrics to JSON file."""
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        print(f"✅ Metrics saved to {filepath}")


def create_tensorboard_callback(dir_name: str, experiment_name: str) -> tf.keras.callbacks.Callback:
    """
    Create a TensorBoard callback.
    
    Args:
        dir_name: Directory to save logs
        experiment_name: Name of the experiment
        
    Returns:
        TensorBoard callback instance
    """
    log_dir = os.path.join(dir_name, experiment_name)
    return tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


def create_training_callbacks(
    model_dir: str = "model_checkpoints",
    tensorboard_dir: str = "training_logs",
    experiment_name: str = "default",
    patience: int = 5,
    reduce_lr_patience: int = 3,
    reduce_lr_factor: float = 0.2,
    min_lr: float = 1e-6
) -> List[tf.keras.callbacks.Callback]:
    """
    Create a comprehensive set of training callbacks.
    
    Args:
        model_dir: Directory for saving model checkpoints
        tensorboard_dir: Directory for TensorBoard logs
        experiment_name: Name of the experiment
        patience: Patience for early stopping
        reduce_lr_patience: Patience for ReduceLROnPlateau
        reduce_lr_factor: Factor to reduce learning rate by
        min_lr: Minimum learning rate threshold
        
    Returns:
        List of Keras callbacks
    """
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    callbacks = []
    
    # ModelCheckpoint
    checkpoint_path = os.path.join(
        model_dir, 
        f"model_epoch_{{epoch:02d}}_val_acc{{val_accuracy:.2f}}.ckpt"
    )
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    ))
    
    # EarlyStopping
    callbacks.append(tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    ))
    
    # ReduceLROnPlateau
    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=reduce_lr_factor,
        patience=reduce_lr_patience,
        min_lr=min_lr,
        verbose=1
    ))
    
    # TensorBoard
    callbacks.append(create_tensorboard_callback(tensorboard_dir, experiment_name))
    
    # Custom logging callback
    def on_epoch_end(epoch, logs):
        if logs:
            print(f"\n📊 Epoch {epoch}: Loss={logs.get('loss', 0):.4f}, "
                  f"Acc={logs.get('accuracy', 0):.4f}, "
                  f"Val_Loss={logs.get('val_loss', 0):.4f}, "
                  f"Val_Acc={logs.get('val_accuracy', 0):.4f}")
    
    callbacks.append(tf.keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end))
    
    return callbacks


def train_model(
    model: Model,
    train_data,
    val_data,
    epochs: int,
    initial_epoch: int = 0,
    callbacks: Optional[List] = None,
    verbose: int = 1
) -> Tuple[Dict, float]:
    """
    Train model with enhanced error handling and monitoring.
    
    Args:
        model: TensorFlow model to train
        train_data: Training dataset
        val_data: Validation dataset
        epochs: Number of epochs to train
        initial_epoch: Starting epoch number
        callbacks: List of callbacks to use
        verbose: Verbosity level (0, 1, or 2)
        
    Returns:
        Tuple of (training history, training time in seconds)
    """
    try:
        start_time = time.time()
        
        print(f"🚀 Starting training from epoch {initial_epoch} to {epochs}...")
        print(f"📅 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if model.optimizer is None:
            raise ValueError("❌ Model is not compiled. Call model.compile() first.")
        
        history = model.fit(
            train_data,
            epochs=epochs,
            initial_epoch=initial_epoch,
            validation_data=val_data,
            callbacks=callbacks if callbacks else [],
            verbose=verbose
        )
        
        training_time = time.time() - start_time
        
        print(f"\n✅ Training completed!")
        print(f"📅 End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  Total training time: {training_time/60:.2f} minutes")
        
        return history, training_time
        
    except Exception as e:
        print(f"❌ Training failed with error: {str(e)}")
        raise


def display_training_plan(
    model: Model,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    train_samples: int,
    val_samples: int
) -> None:
    """
    Display comprehensive training plan before starting.
    
    Args:
        model: Model to train
        epochs: Number of epochs
        learning_rate: Learning rate value
        batch_size: Batch size
        train_samples: Number of training samples
        val_samples: Number of validation samples
    """
    steps_per_epoch = train_samples // batch_size
    val_steps = val_samples // batch_size
    total_steps = steps_per_epoch * epochs
    
    info = get_model_summary_info(model)
    estimated_time_seconds = total_steps * 0.75
    estimated_time_minutes = estimated_time_seconds / 60
    estimated_time_hours = estimated_time_minutes / 60
    
    print("\n" + "🎯 "*30)
    print("TRAINING PLAN")
    print("🎯 "*30)
    print(f"\n📚 DATA:")
    print(f"   Training Samples: {train_samples:,}")
    print(f"   Validation Samples: {val_samples:,}")
    print(f"   Batch Size: {batch_size}")
    
    print(f"\n⚙️ CONFIGURATION:")
    print(f"   Total Epochs: {epochs}")
    print(f"   Steps per Epoch: {steps_per_epoch}")
    print(f"   Validation Steps: {val_steps}")
    print(f"   Total Training Steps: {total_steps:,}")
    print(f"   Learning Rate: {learning_rate}")
    
    print(f"\n🧠 MODEL:")
    print(f"   Total Parameters: {info['total_parameters']:,}")
    print(f"   Trainable Parameters: {info['trainable_parameters']:,}")
    
    print(f"\n⏱️ ESTIMATED TRAINING TIME:")
    if estimated_time_hours > 1:
        print(f"   ~{estimated_time_hours:.2f} hours")
    else:
        print(f"   ~{estimated_time_minutes:.2f} minutes")
    
    print("\n" + "🎯 "*30 + "\n")


def unfreeze_layers(model: Model, num_layers: int = 50) -> None:
    """
    Unfreeze specific number of layers for fine-tuning.
    
    Args:
        model: TensorFlow model
        num_layers: Number of layers to unfreeze from the end
    """
    total_layers = len(model.layers)
    
    for layer in model.layers[:-num_layers]:
        layer.trainable = False
    
    for layer in model.layers[-num_layers:]:
        layer.trainable = True
    
    trainable = sum([1 for layer in model.layers if layer.trainable])
    print(f"✅ Unfrozen {trainable}/{total_layers} layers for fine-tuning")


# ============================================================================
# SECTION 5: EVALUATION & ANALYSIS
# ============================================================================

def get_model_summary_info(model: Model) -> Dict:
    """
    Extract comprehensive model information.
    
    Args:
        model: TensorFlow model
        
    Returns:
        Dictionary with model statistics
    """
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) 
                           for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    return {
        "total_layers": len(model.layers),
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "non_trainable_parameters": int(non_trainable_params),
        "model_name": model.name,
        "input_shape": str(model.input_shape),
        "output_shape": str(model.output_shape)
    }


def print_model_info(model: Model) -> None:
    """Print detailed model information."""
    info = get_model_summary_info(model)
    
    print("\n" + "="*60)
    print("📋 MODEL SUMMARY")
    print("="*60)
    print(f"Model Name: {info['model_name']}")
    print(f"Total Layers: {info['total_layers']}")
    print(f"Input Shape: {info['input_shape']}")
    print(f"Output Shape: {info['output_shape']}")
    print(f"\n📊 Parameters:")
    print(f"   Total Parameters: {info['total_parameters']:,}")
    print(f"   Trainable Parameters: {info['trainable_parameters']:,}")
    print(f"   Non-trainable Parameters: {info['non_trainable_parameters']:,}")
    print(f"\n   Trainable % = {100*info['trainable_parameters']/info['total_parameters']:.2f}%")
    print("="*60 + "\n")


def evaluate_model(
    model: Model,
    test_data,
    batch_size: int = 32,
    verbose: int = 1
) -> Dict[str, float]:
    """
    Evaluate model on test data with detailed metrics.
    
    Args:
        model: Trained TensorFlow model
        test_data: Test dataset
        batch_size: Batch size for evaluation
        verbose: Verbosity level
        
    Returns:
        Dictionary with evaluation metrics
    """
    try:
        print("🔍 Evaluating model on test data...")
        
        results = model.evaluate(test_data, verbose=verbose)
        
        if isinstance(results, (list, tuple)):
            if len(results) == 2:
                loss, accuracy = results
                metrics = {"loss": float(loss), "accuracy": float(accuracy)}
            else:
                metrics = {f"metric_{i}": float(r) for i, r in enumerate(results)}
        else:
            metrics = {"loss": float(results)}
        
        print(f"\n📈 Evaluation Results:")
        for metric_name, metric_value in metrics.items():
            print(f"   {metric_name}: {metric_value:.4f}")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        raise


def compare_models(
    model1: Model,
    model2: Model,
    test_data
) -> Dict:
    """
    Compare two models on test data.
    
    Args:
        model1: First model
        model2: Second model
        test_data: Test dataset
        
    Returns:
        Dictionary with comparison metrics
    """
    print("\n📊 Comparing models...")
    
    results1 = model1.evaluate(test_data, verbose=0)
    results2 = model2.evaluate(test_data, verbose=0)
    
    if isinstance(results1, (list, tuple)):
        loss1, acc1 = results1[0], results1[1]
        loss2, acc2 = results2[0], results2[1]
    else:
        loss1 = results1
        loss2 = results2
        acc1 = acc2 = 0
    
    comparison = {
        "model1_loss": float(loss1),
        "model1_accuracy": float(acc1),
        "model2_loss": float(loss2),
        "model2_accuracy": float(acc2),
        "loss_improvement": float(loss1 - loss2),
        "accuracy_improvement": float(acc2 - acc1),
        "winner": "Model 2" if acc2 > acc1 else "Model 1"
    }
    
    print(f"\n🏆 COMPARISON RESULTS:")
    print(f"   Model 1 - Loss: {comparison['model1_loss']:.4f}, Acc: {comparison['model1_accuracy']:.4f}")
    print(f"   Model 2 - Loss: {comparison['model2_loss']:.4f}, Acc: {comparison['model2_accuracy']:.4f}")
    print(f"   Winner: {comparison['winner']}")
    print(f"   Accuracy Improvement: {comparison['accuracy_improvement']:+.4f}")
    
    return comparison


# ============================================================================
# SECTION 6: VISUALIZATION
# ============================================================================

def plot_loss_curves(history) -> None:
    """
    Plot training and validation loss and accuracy curves.
    
    Args:
        history: Keras History object from model.fit()
    """
    training_loss = history.history['loss']
    val_loss = history.history['val_loss']
    training_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(training_loss))
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, training_loss, label="training_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, training_accuracy, label="training_accuracy")
    plt.plot(epochs, val_accuracy, label="val_accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()


def make_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    classes: Optional[Union[List[str], np.ndarray]] = None, 
    figsize: Tuple[int, int] = (10, 10), 
    text_size: int = 15, 
    norm: bool = False, 
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Generate a labelled confusion matrix plot.
    
    Args:
        y_true: Array of truth labels
        y_pred: Array of predicted labels
        classes: List of class names
        figsize: Figure size
        text_size: Text size in cells
        norm: Whether to normalize the matrix
        save_path: Path to save the figure
        
    Returns:
        Tuple of (figure, axes) objects
    """
    calc_labels = None
    if classes is not None:
        calc_labels = np.arange(len(classes))
        
    cm = confusion_matrix(y_true, y_pred, labels=calc_labels)
    
    if norm:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
        fmt = ".2f"
    else:
        fmt = "d"

    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    if classes is not None:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    ax.set_title("Confusion Matrix", fontsize=text_size + 2, pad=20)
    ax.set_xlabel("Predicted Label", fontsize=text_size)
    ax.set_ylabel("True Label", fontsize=text_size)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(labels)

    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    threshold = (cm.max() + cm.min()) / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        val = cm[i, j]
        color = "white" if val > threshold else "black"
        ax.text(j, i, format(val, fmt),
                horizontalalignment="center",
                verticalalignment="center",
                color=color,
                size=text_size)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")

    return fig, ax


def compare_history(original_history, new_history, initial_epochs: int = 5, 
                   figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Compare two training histories side by side.
    
    Args:
        original_history: History from initial training
        new_history: History from fine-tuning
        initial_epochs: Number of initial epochs
        figsize: Figure size
    """
    orig_acc = original_history.history.get("accuracy", [])
    orig_loss = original_history.history.get("loss", [])
    orig_val_acc = original_history.history.get("val_accuracy", [])
    orig_val_loss = original_history.history.get("val_loss", [])

    new_acc = new_history.history.get("accuracy", [])
    new_loss = new_history.history.get("loss", [])
    new_val_acc = new_history.history.get("val_accuracy", [])
    new_val_loss = new_history.history.get("val_loss", [])

    orig_epochs = range(1, len(orig_acc) + 1)
    new_epochs = range(initial_epochs + 1, initial_epochs + 1 + len(new_acc))

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Training History Comparison: Original vs Fine-tuned', fontsize=16)

    # Original Accuracy
    axes[0, 0].plot(orig_epochs, orig_acc, 'b-', label='Training Accuracy', linewidth=2)
    axes[0, 0].plot(orig_epochs, orig_val_acc, 'r-', label='Validation Accuracy', linewidth=2)
    axes[0, 0].set_title('Original Training - Accuracy')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Fine-tuned Accuracy
    axes[0, 1].plot(new_epochs, new_acc, 'g-', label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(new_epochs, new_val_acc, 'orange', label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_title('Fine-tuned Training - Accuracy')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Original Loss
    axes[1, 0].plot(orig_epochs, orig_loss, 'b-', label='Training Loss', linewidth=2)
    axes[1, 0].plot(orig_epochs, orig_val_loss, 'r-', label='Validation Loss', linewidth=2)
    axes[1, 0].set_title('Original Training - Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Fine-tuned Loss
    axes[1, 1].plot(new_epochs, new_loss, 'g-', label='Training Loss', linewidth=2)
    axes[1, 1].plot(new_epochs, new_val_loss, 'orange', label='Validation Loss', linewidth=2)
    axes[1, 1].set_title('Fine-tuned Training - Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Original Training:")
    print(f"  • Epochs: {len(orig_acc)}")
    print(f"  • Final Training Accuracy: {orig_acc[-1]:.4f}")
    print(f"  • Final Validation Accuracy: {orig_val_acc[-1]:.4f}")
    print(f"  • Final Training Loss: {orig_loss[-1]:.4f}")
    print(f"  • Final Validation Loss: {orig_val_loss[-1]:.4f}")
    
    print(f"\nFine-tuning:")
    print(f"  • Epochs: {len(new_acc)}")
    print(f"  • Final Training Accuracy: {new_acc[-1]:.4f}")
    print(f"  • Final Validation Accuracy: {new_val_acc[-1]:.4f}")
    print(f"  • Final Training Loss: {new_loss[-1]:.4f}")
    print(f"  • Final Validation Loss: {new_val_loss[-1]:.4f}")
    
    acc_improvement = new_val_acc[-1] - orig_val_acc[-1]
    loss_improvement = orig_val_loss[-1] - new_val_loss[-1]
    print(f"\nImprovement from Fine-tuning:")
    print(f"  • Validation Accuracy: {acc_improvement:+.4f}")
    print(f"  • Validation Loss: {loss_improvement:+.4f}")
    print("="*60)


def plot_metric_scores(
    df: pd.DataFrame, 
    class_col: str, 
    score_col: str, 
    title: str = "Class Performance",
    xlabel: str = "Score",
    figsize: Tuple[int, int] = (12, 10),
    color: str = "#1f77b4",
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Generate a horizontal bar chart for classification metrics.
    
    Args:
        df: DataFrame with data
        class_col: Column name for class labels
        score_col: Column name for scores
        title: Chart title
        xlabel: X-axis label
        figsize: Figure size
        color: Bar color
        save_path: Path to save figure
        
    Returns:
        Tuple of (figure, axes)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = range(len(df))
    bars = ax.barh(y_pos, df[score_col], color=color, align='center')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df[class_col])
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.invert_yaxis()

    x_offset = df[score_col].max() * 0.01 

    for rect in bars:
        width = rect.get_width()
        label_x_pos = width + x_offset
        ax.text(label_x_pos, rect.get_y() + rect.get_height() / 2, f"{width:.2f}", 
                ha='left', va='center', fontsize=10, color='black')

    ax.set_xlim(right=df[score_col].max() * 1.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to {save_path}")

    return fig, ax


# ============================================================================
# SECTION 7: METRICS CALCULATION
# ============================================================================

def calculate_results(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate accuracy, precision, recall and F1-score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with metrics
    """
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1 = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


def pred_and_plot(model: Model, filename: str, class_names: List[str]) -> None:
    """
    Make prediction on an image and display it.
    
    Args:
        model: Trained model
        filename: Path to image file
        class_names: List of class names
    """
    img = ImageProcessor.load_and_prep_image(filename)
    pred = model.predict(tf.expand_dims(img, axis=0))

    if len(pred[0]) > 1:
        pred_class = class_names[tf.argmax(pred[0])]
    else:
        pred_class = class_names[tf.round(pred)]

    plt.imshow(img)
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False)


# ============================================================================
# SECTION 8: CONFIGURATION MANAGEMENT
# ============================================================================

def save_training_config(config: Dict, save_path: str = "training_config.json") -> None:
    """Save training configuration to JSON file."""
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"✅ Training config saved to {save_path}")


def load_training_config(filepath: str) -> Dict:
    """Load training configuration from file."""
    with open(filepath, 'r') as f:
        config = json.load(f)
    print(f"✅ Training config loaded from {filepath}")
    return config