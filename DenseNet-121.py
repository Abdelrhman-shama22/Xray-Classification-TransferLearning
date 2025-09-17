import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

print("TensorFlow version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices('GPU'))

train_dir = r'C:\abdo123\chest_xray\train'
test_dir = r'C:\abdo123\chest_xray\test'

img_size = (224, 224)
batch_size = 32
seed = 1337

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    color_mode='rgb',
    subset='training',
    seed=seed
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    color_mode='rgb',
    subset='validation',
    seed=seed
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    color_mode='rgb',
    shuffle=False
)

classes = train_generator.classes
unique, counts = np.unique(classes, return_counts=True)
count_dict = {int(k): int(v) for k, v in zip(unique, counts)}
total = sum(counts)
class_weight = {c: total/(2.0*count_dict[c]) for c in count_dict}
print("Class counts:", count_dict, " -> class_weight:", class_weight)

base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))
base_model.trainable = False

inputs = layers.Input(shape=(img_size[0], img_size[1], 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=BinaryCrossentropy(label_smoothing=0.05),
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

model.summary()


ckpt_path = "best_densenet_chestxray.keras" 
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6, verbose=1),
    
    ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True, verbose=1)  
]

print("\n--- Training the head ---")
history_head = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)

print("\n--- Fine-tuning the model ---")
base_model.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss=BinaryCrossentropy(label_smoothing=0.05),
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

history_fine_tune = model.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)

print("\nEvaluating on the final test set...")
test_loss, test_acc, test_auc = model.evaluate(test_generator, verbose=1)

print(f"\n✅ Final Test Accuracy: {test_acc * 100:.2f}%")
print(f"✅ Final Test AUC: {test_auc:.4f}")

final_model_path = "final_densenet_chestxray.keras"
model.save(final_model_path)
print(f"\n Model saved successfully to: {final_model_path}")




print("\nCalculating best threshold...")

y_pred_prob = model.predict(test_generator).ravel()
y_true = test_generator.classes

precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_prob)

f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print(f"Best Threshold: {best_threshold:.2f}")
print(f"Best F1-score: {best_f1:.2f}")

plt.plot(recalls, precisions, label="PR Curve")
plt.scatter(recalls[best_idx], precisions[best_idx], color='red', label=f"Best Th={best_threshold:.2f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.show()
