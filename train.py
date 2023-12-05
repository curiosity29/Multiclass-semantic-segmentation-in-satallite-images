import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import numpy
import numpy as np
import tensorflow as tf
import segmentation_models as sm

from matplotlib import pyplot as plt
from proposed_model import multi_unet_model, jacard_coef
from patchify import patchify
from PIL import Image

#import segmentation_models as sm
from Load_data import get_dataset
from set_labels import set_labels 

if __name__ == '__main__':
    
    # Path
    data_folder = "/content/drive/MyDrive/Colab Notebooks/Building_footprint/MultiClass_data"
    checkpoint_filepath_save = "/content/drive/MyDrive/Colab Notebooks/Building_footprint/CheckPoint/MS2/trained_model.hdf5"
    checkpoint_filepath_load = "/content/drive/MyDrive/Colab Notebooks/Building_footprint/CheckPoint/MS2/trained_model.hdf5"
    #

    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    image_dataset, mask_dataset = get_dataset(data_folder)
    print(image_dataset.shape[0])
    print(mask_dataset.shape[0])

    # Set labels
    # labels_cat = set_labels(mask_dataset)

    import random
    import numpy as np

    image_number = random.randint(0, len(image_dataset))
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(image_dataset[image_number])
    plt.subplot(122)
    plt.imshow(mask_dataset[image_number])
    plt.show()

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size=0.3, random_state=42)

    # Loss functions setting
    weights = [0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666]
    dice_loss = sm.losses.DiceLoss(class_weights=weights, ignore_index = 0)
    focal_loss = sm.losses.CategoricalFocalLoss(ignore_index = 0)
    total_loss = dice_loss + (1 * focal_loss)

    # Load model
    IMG_HEIGHT = X_train.shape[1]
    IMG_WIDTH = X_train.shape[2]
    IMG_CHANNELS = X_train.shape[3]

    metrics = ['accuracy', jacard_coef]


    def get_model():
        return multi_unet_model(n_classes=8, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH,
                                IMG_CHANNELS=IMG_CHANNELS)


    model = get_model()
    model.summary()



    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath_load,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    callback_list = [model_checkpoint_callback]

    model.compile(optimizer='adam', loss=total_loss, metrics=metrics)
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)
    model.summary()

    history1 = model.fit(X_train, y_train,
                         batch_size=8,
                         verbose=1,
                         epochs=50,
                         validation_data=(X_test, y_test),
                         shuffle=True,
                         callbacks=callback_list)
    model.save(checkpoint_filepath_save)











