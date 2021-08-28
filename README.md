# NeuralNetCodeSnippets
🧠 A set of useful code snippets for neural network applications
## Making models in Keras
### Creating a CNN
```
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu, input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10,  activation=tf.nn.softmax)
])
```

### Creating a LSTM
```
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```
### Creating a RNN
```
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

### Creating a RNN with attention
```
from attention import AttentionLayer

model = tf.keras.models.Sequential()
model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True),
                               input_shape=(30, 300)))
model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
model.add(AttentionLayer(300))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
```

##Compiling, Training and Evaluating in Keras
### Compiling a model
```
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

### Training a model
```
#Train the model
model.fit(x_train, y_train, epochs=5)
```

### Evaluating a model
```
test_loss, test_acc = model.evaluate(x_test, y_test)

print('Test accuracy:', test_acc)
```
