import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog



MODEL_PATH = 'final_densenet_chestxray.keras'

IMG_SIZE = (224, 224)


THRESHOLD =  0.74

CLASS_LABELS = {0: 'NORMAL', 1: 'PNEUMONIA'}


try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model '{MODEL_PATH}' loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

def predict_single_image(img_path, model, threshold):
    """
    تقوم هذه الدالة بتحميل صورة، تجهيزها، التنبؤ بنتيجتها، وعرضها.
    """
    img = image.load_img(img_path, target_size=IMG_SIZE, color_mode='rgb')
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    prediction_prob = model.predict(img_batch)[0][0]

    if prediction_prob >= threshold:
        final_prediction_label = CLASS_LABELS[1]
    else:
        final_prediction_label = CLASS_LABELS[0]

    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Prediction: {final_prediction_label}\nProbability: {prediction_prob:.4f}")
    plt.show()

    print(f"Image Path: {img_path}")
    print(f"Prediction Probability (raw score): {prediction_prob:.4f}")
    print(f"Threshold: {THRESHOLD}")
    print(f"Final Decision: {final_prediction_label}")


if __name__ == '__main__':
    # tkinter 
    root = tk.Tk()
    root.withdraw() 
    
    print("\nPlease select an image file to test...")
    
    
    image_path = filedialog.askopenfilename(
        title="Select a Chest X-ray Image",
        filetypes=[("Image Files", "*.jpeg *.jpg *.png *.bmp")]
    )

    
    if image_path:
       
        predict_single_image(image_path, model, THRESHOLD)
    else:
       
        print("\nNo image selected. Exiting.")

