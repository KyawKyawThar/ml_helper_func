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
    for dirnames,dirpath,filenames in os.walk(dir_path):
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
    plt.subplots(1,2,1)
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

# Constants
# ==============================================
# 1. Define the Custom FeatureExtractor Layer
# ==============================================
class FeatureExtractor(tf.keras.layers.Layer):
    """
    Custom layer to extract features from an input tensor.
    
    Standalone reusable layer for TF Hub feature extraction.
    """
    def __init__(self,model_url,image_shape,trainable=False,**kwargs):
        super().__init__(**kwargs)
        self.model_url = model_url
        self.trainable = trainable
        self.image_shape = image_shape

    # Load the TF Hub module
        self.feature_extractor = hub.KerasLayer(model_url,
                                            trainable=trainable,
                                            input_shape = image_shape + (3,))
    def call(self, inputs):
        """Normalize inputs and extract features."""
        # Convert to float32 and scale to [0,1]
        x = tf.cast(inputs, tf.float32) / 255.0  
        return self.feature_extractor(x)
    
    def get_config(self):
        """For model serialization."""
        config = super().get_config()
        config.update({"model_url": self.model_url, "trainable": self.trainable})
        return config

# ==============================================
# 2. Model Creation Function
# ==============================================

def create_model(model_url,image_shape, num_classes=10):
    """
    Creates a model with a custom FeatureExtractor layer.
    
    Args:
        model_url (str): TF Hub model URL
        num_classes (int): Number of output classes
    
    Returns:
        tf.keras.Model: Compiled model
    """
    # Input layer
    inputs = tf.keras.Input(shape=image_shape + (3,))
    
    # Feature extraction
    features = FeatureExtractor(model_url,image_shape, trainable=False)(inputs)
    
    # Classification head
    outputs = layers.Dense(num_classes, activation='softmax')(features)
    
    # Build model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model

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
