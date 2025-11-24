import os


def walk_through_directory(dir_path):
    """
    Walks through dir_path returning its contents.

  Args:
    dir_path (str): target directory
  
  Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
"""
    for dirpath,dirnames,filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

    
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random


def view_random_image(target_dir,target_class):
    """
    Displays a random image from the specified class directory.

  Args:
    target_dir (str): path to the target directory
    target_class (str): name of the class directory
  
  Returns:
    image
"""
    target_folder = target_dir + target_class
    random_image = random.sample(os.listdir(target_folder),1)
    print(f"Displaying random image from {target_folder}: {random_image}")
    img = mpimg.imread(target_folder + '/' + random_image[0])
    plt.imshow(img)
    plt.axis('off')
    plt.title(target_class)


    return img


def plot_loss_curves(history):
    """
    Returns separate loss curves for training and validation metrics.
    Args:
        history: A Keras History object (returned from model.fit()).
    """
    training_loss = history.history['loss']
    val_loss = history.history['val_loss']

    training_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(training_loss)) # how many epochs did we run for
    plt.figure(figsize=(12,5))

    ## plot loss
    plt.subplot(1,2,1)
    plt.plot(epochs,training_loss,label="training_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    ## plot accuracy
    plt.subplot(1,2,2)
    plt.plot(epochs,training_accuracy, label="training_accuracy")
    plt.plot(epochs, val_accuracy, label="val_accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()


import tensorflow as tf

def load_and_prep_image(filename,image_shape=224):
    """
    Read an images from filename,turns it into a tensor and reshape it to
    (img_shape,img_shape,color_channels)
    """
    # Read the image from file
    img =tf.io.read_file(filename)
    # Decode the image into a tensor
    img = tf.image.decode_image(img,channels=3)

    # Resize the image
    img = tf.image.resize(img,size=[image_shape,image_shape])
    # Rescale the image (get all values between 0 and 1)
    img = img/255
    return img


def pred_and_plot(model,filename,class_names):
    """
     Imports an image located at filename, make a predictions with model and plots
    the images with the predicted class as the title
    """
    img = load_and_prep_image(filename)
    #Make a predictions
    pred = model.predict(tf.expand_dims(img,axis=0))
    print("pred\n", pred)
    print("length\n", len(pred[0]))
    print("argmax\n", tf.argmax(pred))
    print("lenPred\n",tf.argmax(pred[0]))

    # Get the predicted class
    if len(pred[0]) > 1:
        pred_class = class_names[tf.argmax(pred[0])] # if more than one output, take the max
    else:
        pred_class = class_names[tf.round(pred)] # if only one output, round

    # Plot the image and predict class
    plt.imshow(img)
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False)

# Create tensorboard callback (functionized because need to create a new one for each model)
import tensorflow as tf
def create_tensorboard_callback(dir_name, experiment_name):
    """
    Creates a TensorBoard callback to log training metrics.
    
    Args:
        dir_name (str): Directory to save the logs.
        experiment_name (str): Name of the experiment.
    
    Returns:
        A TensorBoard callback instance.
    """
    log_dir = os.path.join(dir_name, experiment_name)
    return tf.keras.callbacks.TensorBoard(log_dir=log_dir)


import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow.keras import Sequential


data_augmention=Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomHeight(0.2),
    layers.RandomWidth(0.2)
], name="data_augmentation")

# ==============================================
# 1. Model Creation Function
# ==============================================

def create_model(base_model,image_shape, num_classes=10):
    """
    Creates a model with a custom FeatureExtractor layer.
    
    You shouldn't always use 1./255 (which scales data to [0, 1]) because different pre-trained models were trained on different data ranges.
    If you feed a model data in the range [0, 1] when it expects [-1, 1], the model's mathematical weights will react incorrectly, 
    and your accuracy will likely drop significantly (or the model won't learn at all).

    When researchers train models (like Google with EfficientNet or Microsoft with ResNet),
    they pick a specific way to "normalize" images before training. You must match that choice exactly.

     Range	     Formula	                       Common Models
     [0, 1]	 x / 255	                       Many custom models, TF Hub models, DenseNet.
     [-1, 1]	(x - 127.5) / 127.5	               ResNetV2, MobileNetV2, InceptionV3, Xception.
     Unscaled  [0, 255]No scaling (Raw pixels)	   EfficientNetV2, EfficientNet (B0-B7).
     Caffe Style	x - Mean_RGB (keep 0-255)	   VGG16, ResNet50 (V1).

     Why EfficientNetV2 is special
     EfficientNetV2 (and V1) was designed to be user-friendly. The Google team included a Rescaling layer inside the model architecture itself.
     If you do: inputs / 255.0 -> Model internal scaling inputs / 255.0 -> Result: You divided by 255 twice. Your pixel values become tiny (e.g., 0.00001), and the model sees "black" images.
     Correct way: Pass raw [0, 255] data. The model handles the math internally.

    Why ResNetV2 and MobileNet use [-1, 1]
    Mathematical "centering" helps models train faster.
    [0, 1] means all inputs are positive.
    [-1, 1] means the average input is roughly 0.
    Neural networks generally converge faster and more stably when the input data is centered around zero. Since ResNetV2 expects inputs between -1 and 1:
    If you give it [0, 1], you are only using the positive half of the range it expects.
    The pre-trained weights (which are fixed) will fire activations that are weaker than they should be.

    How to calculate the [-1, 1] formula?
    If you want to convert [0, 255] to [-1, 1], you cannot just divide.
    Step 1: x / 127.5 
    Range becomes [0, 2]
    Step 2: Subtract 1 
    Range becomes [-1, 1]
    Equivalent to (x - 127.5) / 127.5
    layers.Rescaling(scale=1./127.5, offset=-1)
    
   Summary Checklist
   When using tf.keras.applications:
   EfficientNetV2: Do NOT scale. Input [0, 255].
   ResNetV2 / MobileNetV2: Scale to [-1, 1].
   VGG16: Use tf.keras.applications.vgg16.preprocess_input.
   TF Hub Models (URL strings): Usually scale to [0, 1]
    Args:
        base_model (model_object): TF Hub model URL
        num_classes (int): Number of output classes
    
    Returns:
        tf.keras.Model: Compiled model
    """
    # 1. Input layer
    inputs = tf.keras.Input(shape=image_shape + (3,))

    # 2. Augmentation (operates on [0-255] range)
    X = data_augmention(inputs)
    
    # 3. Dynamic Preprocessing Logic
    # We check the name of the model object to decide how to scale.
    model_name = base_model.name.lower()
  
    if "efficientnet" in model_name:
        # EfficientNet models expect inputs in the range [0, 255]
        print(f"Model: {base_model.name} -> Skipping external Rescaling (uses [0, 255]).")
        pass
    elif "resnet" in model_name and "v2" in model_name:
         # ResNetV2 expects inputs in range [-1, 1].
         # We convert from [0, 255] to [-1, 1].
         print(f"Model: {base_model.name} -> Adding Rescaling [-1, 1].")
         X = layers.Rescaling(1./127.5, offset=-1)(X)
    elif "mobilenet" in model_name:
         # MobileNetV2 usually expects [-1, 1]
         print(f"Model: {base_model.name} -> Adding Rescaling [-1, 1].")
         X = layers.Rescaling(1./127.5, offset=-1)(x)
    else:
        # Fallback: Standard Rescaling [0, 1] for VGG16, DenseNet, etc.
        print(f"Model: {base_model.name} -> Defaulting to Rescaling [0, 1].")
        X = layers.Rescaling(1./255.0)(X)

    # 4. Feature Extraction
    # Set training=False to keep BatchNormalization layers in inference mode
    # even if we unfreeze weights later.

    X = base_model(X,training=False)
    
    # 5. Pooling (REQUIRED for include_top=False)
    # Converts 4D tensor (batch, h, w, c) -> 2D tensor (batch, c)
    X = layers.GlobalAveragePooling2D()(X)

    # Classification head
    outputs = layers.Dense(num_classes, activation='softmax')(X)
    
    # Build model
    return tf.keras.Model(inputs=inputs, outputs=outputs)
    
   

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import numpy as np

def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False): 
    """Makes a labelled confusion matrix comparing predictions and ground truth labels.

  If classes is passed, confusion matrix will be labelled, if not, integer class values
  will be used.

  Args:
    y_true: Array of truth labels (must be same shape as y_pred).
    y_pred: Array of predicted labels (must be same shape as y_true).
    classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
    figsize: Size of output figure (default=(10, 10)).
    text_size: Size of output figure text (default=15).
    norm: normalize values or not (default=False).
    savefig: save confusion matrix to file (default=False).
  
  Returns:
    A labelled confusion matrix plot comparing y_true and y_pred.

  Example usage:
    make_confusion_matrix(y_true=test_labels, # ground truth test labels
                          y_pred=y_preds, # predicted labels
                          classes=class_names, # array of class label names
                          figsize=(15, 15),
                          text_size=10)
  """  
    # Create the confustion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Normalize the confusion matrix if norm is True
    if norm:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    n_classes = cm.shape[0] # find the number of classes we're dealing with

    # Plot the figure and make it pretty
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues) # colors will represent how 'correct' a class is, darker == better
    fig.colorbar(cax)
   

    # Are there a list of classes?
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])
     # Label the axes
    ax.set(title="Confusion Matrix",
            xlabel="Predicted label",
            ylabel="True label",
            xticks=np.arange(n_classes), # create enough axis slots for each class
            yticks=np.arange(n_classes), 
            xticklabels=labels, # axes will labeled with class names (if they exist) or ints
            yticklabels=labels)
   # Make x-axis labels appear on bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()
    # Set the threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if norm:
            plt.text(j, i, f"{cm[i, j]}",  # cm_norm is not defined, so just show cm[i, j]
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black",
                     size=text_size)
        else:
            plt.text(j, i, f"{cm[i, j]}",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black",
                     size=text_size)
    # Save the figure to the current working directory
    if savefig:
        fig.savefig("confusion_matrix.png")

def compare_historys(original_history, new_history, initial_epochs=5):
    """
    Compares two training histories by plotting their loss and accuracy curves.
    
    Args:
        original_history: History object from the original model.
        new_history: History object from the new model.
        initial_epochs: Number of epochs to consider for comparison.
    """
    # Get original history measurements
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Combine original history with new history
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    # Make plots
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

# Create function to unzip a zipfile into current working directory 
# (since we're going to be downloading and unzipping a few files)
import zipfile

def unzip_data(filename):
  """
  Unzips filename into the current working directory.

  Args:
    filename (str): a filepath to a target zip folder to be unzipped.
  """
  zip_ref = zipfile.ZipFile(filename, "r")
  zip_ref.extractall()
  zip_ref.close()


# Function to evaluate: accuracy, precision, recall, f1-score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def calculate_results(y_true, y_pred):
    """
  Calculates model accuracy, precision, recall and f1 score of a binary classification model.

  Args:
      y_true: true labels in the form of a 1D array
      y_pred: predicted labels in the form of a 1D array

  Returns a dictionary of accuracy, precision, recall, f1-score.
  """
    # Calculate accuracy
    acc = accuracy_score(y_true, y_pred)

    # Calculate precision, recall and f1-score
    precision, recall, f1 = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    # Return results as a dictionary
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


def compare_history(original_history,new_history,initial_epochs=5,figsize=(15,10)):
    """
    Compare model training histories side by side.
    
    Args:
        original_history: History object from initial training
        new_history: History object from fine-tuning
        initial_epochs: Number of initial training epochs
        figsize: Figure size tuple
    """
    # Extract original history
    orig_acc = original_history.history.get("accuracy", [])
    orig_loss = original_history.history.get("loss", [])
    orig_val_acc = original_history.history.get("val_accuracy", [])
    orig_val_loss = original_history.history.get("val_loss", [])

    # Extract new history
    new_acc = new_history.history.get("accuracy", [])
    new_loss = new_history.history.get("loss", [])
    new_val_acc = new_history.history.get("val_accuracy", [])
    new_val_loss = new_history.history.get("val_loss", [])

    # Create epochs arrays
    orig_epochs = range(1, len(orig_acc) + 1)
    new_epochs = range(initial_epochs + 1, initial_epochs + 1 + len(new_acc))

    # Create side-by-side plots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Training History Comparison: Original vs Fine-tuned', fontsize=16)

    # Original Training - Accuracy
    axes[0, 0].plot(orig_epochs, orig_acc, 'b-', label='Training Accuracy', linewidth=2)
    axes[0, 0].plot(orig_epochs, orig_val_acc, 'r-', label='Validation Accuracy', linewidth=2)
    axes[0, 0].set_title('Original Training - Accuracy')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Fine-tuned Training - Accuracy
    axes[0, 1].plot(new_epochs, new_acc, 'g-', label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(new_epochs, new_val_acc, 'orange', label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_title('Fine-tuned Training - Accuracy')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Original Training - Loss
    axes[1, 0].plot(orig_epochs, orig_loss, 'b-', label='Training Loss', linewidth=2)
    axes[1, 0].plot(orig_epochs, orig_val_loss, 'r-', label='Validation Loss', linewidth=2)
    axes[1, 0].set_title('Original Training - Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Fine-tuned Training - Loss
    axes[1, 1].plot(new_epochs, new_loss, 'g-', label='Training Loss', linewidth=2)
    axes[1, 1].plot(new_epochs, new_val_loss, 'orange', label='Validation Loss', linewidth=2)
    axes[1, 1].set_title('Fine-tuned Training - Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
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
    
    # Calculate improvements
    acc_improvement = new_val_acc[-1] - orig_val_acc[-1]
    loss_improvement = orig_val_loss[-1] - new_val_loss[-1]
    print(f"\nImprovement from Fine-tuning:")
    print(f"  • Validation Accuracy: {acc_improvement:+.4f}")
    print(f"  • Validation Loss: {loss_improvement:+.4f}")
    print("="*60)