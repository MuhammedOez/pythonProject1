import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Laden Sie das CIFAR-100-Datensatz
(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.cifar100.load_data()

#train_data = train_data.reshape(len(train_data),3224224)
#test_data = test_data.reshape(len(test_data),3224224)

train_data = train_data.reshape(len(train_data), 32 * 32 * 3)
test_data = test_data.reshape(len(test_data), 32 * 32 * 3)


# Normalisieren Sie die Bilddaten auf die Werte [0, 1]
scaler = StandardScaler().fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)
#x_train = x_train / 255.0
#x_test = x_test / 255.0

train_data = train_data.reshape(len(train_data), 32, 32, 3)
test_data = test_data.reshape(len(test_data), 32, 32, 3)

# One-hot encode the labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Erstellen Sie das CNN-Modell
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='sigmoid', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='sigmoid'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='sigmoid'),
    tf.keras.layers.Dense(100, activation='softmax')
])

# Kompilieren Sie das Modell
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Passen Sie das Modell an, indem Sie die Trainingsdaten verwenden
model.fit(train_data, train_labels, epochs=10)

# Bewerten Sie das Modell auf den Testdaten
test_loss, test_acc = model.evaluate(train_data, train_labels)

print('Test accuracy:', test_acc)
