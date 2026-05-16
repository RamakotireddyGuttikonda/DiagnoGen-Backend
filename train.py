from data_preprocessing import tf_dataset
from data_preprocessing import load_dataset
from brainTumorModel import build_unet
import tensorflow
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from metrics import dice_coef
from metrics import dice_loss

dataset_path = "" # change to your folder
(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)

train_dataset = tf_dataset(train_x, train_y, batch=8)
valid_dataset = tf_dataset(valid_x, valid_y, batch=8)

# Step 8: Build and train the model
model = build_unet()
model.compile(loss=dice_loss, optimizer=Adam(1e-4), metrics=[dice_coef])
model.summary()

model_path = "brain_tumor_seg_model.keras"
callbacks = [
    ModelCheckpoint(model_path, save_best_only=True, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-7, verbose=1),
    EarlyStopping(patience=15, monitor="val_loss", verbose=1)
]

history = model.fit(train_dataset, validation_data=valid_dataset, epochs=35, callbacks=callbacks)