import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

print("TensorFlow 버전:", tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = create_model()
start_time = time.time()
model.fit(x_train[:1000], y_train[:1000], epochs=2, batch_size=32, verbose=1)
end_time = time.time()

print(f"\n완료 시간: {end_time - start_time:.2f}초")
test_loss, test_acc = model.evaluate(x_test[:100], y_test[:100], verbose=2)
print(f"정확도: {test_acc:.4f}")
predictions = model.predict(x_test[:5])
plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"예측: {np.argmax(predictions[i])}")
    plt.axis('off')

plt.tight_layout()
plt.savefig('tensorflow_predictions.png') 