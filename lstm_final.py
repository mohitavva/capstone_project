import os
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator



# Define parameters
frame_size = (224, 224, 3) 
sequence_length = 6*30
num_classes = 16
base_dir = '/mnt/d/Capstone/train_trimmed/train'



base_model = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=frame_size)
cnn = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization()
])


inputs = layers.Input(shape=(sequence_length,) + frame_size)
cnn_outputs = layers.TimeDistributed(cnn)(inputs)
# x = layers.LSTM(256, return_sequences=True)(cnn_outputs)
# x = layers.LSTM(128)(x)
# x = layers.Dense(512, activation='swish')(x)
# x = layers.Dropout(0.3)(x)
# x = layers.BatchNormalization()(x)
x = layers.Dense(num_classes, activation='softmax')(cnn_outputs)
model = models.Model(inputs=inputs, outputs=x)


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])




model.summary()




# X, y = load_data(base_dir)




# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


class SequenceGenerator(Sequence):
    def __init__(self, directory, sequence_length, target_size, batch_size, num_classes=None):

        self.directory = directory
        self.sequence_length = sequence_length
        self.target_size = target_size
        self.batch_size = batch_size
        
        self.class_names = sorted(os.listdir(directory))
        if num_classes:
            self.class_names = self.class_names[:num_classes]  
        self.class_indices = {cls: i for i, cls in enumerate(self.class_names)}
        
        # print(f"Class names: {self.class_names}")
        
        self.samples = self._get_samples()

    def _get_samples(self):
        samples = []
        
        for class_name in self.class_names:
            class_dir = os.path.join(self.directory, class_name)
            
        
            if not os.path.isdir(class_dir):
                print(f"Skipping {class_name}, directory not found.")
                continue
            
        
            for user_name in os.listdir(class_dir):
                user_dir = os.path.join(class_dir, user_name)
                
        
                if not os.path.isdir(user_dir):
                    print(f"Skipping {user_name} in {class_name}, directory not found.")
                    continue
                
        
                frame_files = sorted(os.listdir(user_dir))  
                
                
                print(f"Class {class_name}, User {user_name} has {len(frame_files)} frames.")
                
                for i in range(len(frame_files) - self.sequence_length + 1):
                    samples.append((class_name, user_name, frame_files[i:i + self.sequence_length]))
        
        print(f"Total samples generated: {len(samples)}")
        
        return samples

    def _load_frame(self, frame_path):

        img = load_img(frame_path, target_size=self.target_size)
        img_array = img_to_array(img) / 255.0 
        return img_array

    def __len__(self):
        # Total number of batches per epoch
        return len(self.samples) // self.batch_size

    def __getitem__(self, index):
        # Batch of samples
        batch_samples = self.samples[index * self.batch_size: (index + 1) * self.batch_size]
        print(f"Batch {index}: {len(batch_samples)} samples")
        
        X_batch = []
        y_batch = []

        for class_name, user_name, frames in batch_samples:
            # Load the sequence of frames
            user_dir = os.path.join(self.directory, class_name, user_name)
            X_sequence = np.array([self._load_frame(os.path.join(user_dir, frame)) for frame in frames])
            X_batch.append(X_sequence)
            y_batch.append(self.class_indices[class_name])  # Get the class index for label

        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)

        return X_batch, y_batch

sequence_length = 180
target_size = (224, 224)
batch_size = 32

train_generator = SequenceGenerator(base_dir, sequence_length, target_size, batch_size, num_classes)

train_generator[1]


model.fit(train_generator, epochs=10, steps_per_epoch=len(train_generator))


