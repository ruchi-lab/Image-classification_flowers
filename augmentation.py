# After your existing imports, add:
import os
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# Create directory for augmented images
def create_augmented_dir(base_dir, class_name):
    aug_dir = os.path.join('augmented_flowers', class_name)
    os.makedirs(aug_dir, exist_ok=True)
    return aug_dir

def augment_and_save_images(input_dir, num_augmented_per_image=5):
    # Create different augmentation generators for each type of transformation
    augmenters = {
        'rotation': ImageDataGenerator(rotation_range=45),
        'flip': ImageDataGenerator(horizontal_flip=True, vertical_flip=True),
        'zoom': ImageDataGenerator(zoom_range=[0.5, 1.5]),
        'shift': ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2),
        'brightness': ImageDataGenerator(brightness_range=[0.2, 1.8]),
        'shear': ImageDataGenerator(shear_range=0.3)
    }
    
    # Process each class directory
    for class_name in os.listdir(input_dir):
        class_dir = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        print(f"Processing class: {class_name}")
        aug_dir = create_augmented_dir('augmented_flowers', class_name)
        
        # Process each image in the class directory
        for img_name in os.listdir(class_dir):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            # Load and preprocess the image
            img_path = os.path.join(class_dir, img_name)
            img = load_img(img_path, target_size=(224, 224))
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)
            
            # Generate and save augmented images for each transformation type
            for aug_type, generator in augmenters.items():
                for i in range(num_augmented_per_image):
                    # Get augmented image
                    aug_iter = generator.flow(x, batch_size=1)
                    aug_img = next(aug_iter)[0].astype(np.uint8)
                    
                    # Save the augmented image
                    output_filename = f"{os.path.splitext(img_name)[0]}_{aug_type}_{i}.jpg"
                    output_path = os.path.join(aug_dir, output_filename)
                    Image.fromarray(aug_img).save(output_path)
                    
            print(f"Processed {img_name}")

# Directory containing the original images
input_directory = 'flowers/training'

# Generate augmented images
print("Starting image augmentation...")
augment_and_save_images(input_directory, num_augmented_per_image=3)
print("Augmentation complete!")