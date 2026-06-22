import tensorflow as tf
import numpy as np
import os

# 1. Hardcode classes to match training folders
class_names = ['100', '1000', '200', '2000', '50', '500']

# 2. Boot up and load the model ONCE
print("Loading trained model into memory (this takes a moment)...")
model = tf.keras.models.load_model("algerian_money_model.keras")
print("Model loaded! You can now make instant predictions.\n")

# 3. Enter the continuous interactive loop
while True:
    # Ask the user for an image path
    image_path = input("Enter image filename (or type 'no' to exit): ").strip()
    
    # Check for exit condition
    if image_path.lower() == 'no':
        print("Exiting...")
        break
        
    # Check if the file actually exists so the script doesn't crash on typos
    if not os.path.exists(image_path):
        print(f"❌ Error: File '{image_path}' not found. Please try again.\n")
        continue

    try:
        # 4. Load and process the specific image inside the loop
        img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch dimension

        # 5. Predict instantly!
        predictions = model.predict(img_array, verbose=0)  # verbose=0 hides the loading bar
        score = tf.nn.softmax(predictions[0])

        predicted_class = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)

        print("\n--- RESULT ---")
        print(f"This looks like: {predicted_class} DA")
        print(f"Confidence: {confidence:.2f}%")
        print("--------------\n")
        
    except Exception as e:
        print(f"❌ Failed to process image: {e}\n")