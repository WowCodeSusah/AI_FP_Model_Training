import warnings

warnings.filterwarnings('ignore')

import numpy as np
import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras import layers
import transformer
import tensorflow as tf
import matplotlib.pyplot as plt

def CNN(x_train, y_train, x_test, y_test):
    maximum_features = 5000  # Maximum number of words to consider as features
    maximum_length = 100  # Maximum length of input sequences
    word_embedding_dims = 50  # Dimension of word embeddings
    no_of_filters = 250  # Number of filters in the convolutional layer
    kernel_size = 3  # Size of the convolutional filters
    hidden_dims = 250  # Number of neurons in the hidden layer
    batch_size = 32  # Batch size for training
    epochs = 50  # Number of training epochs
    threshold = 0.5  # Threshold for binary classification

    # Building the model
    model = keras.Sequential()

    # Adding the embedding layer to convert input sequences to dense vectors
    model.add(keras.layers.Embedding(maximum_features, word_embedding_dims, input_length=maximum_length))

    # Adding the 1D convolutional layer with ReLU activation
    model.add(keras.layers.Conv1D(no_of_filters, kernel_size, padding='valid', activation='relu', strides=1))

    # Adding the global max pooling layer to reduce dimensionality
    model.add(keras.layers.GlobalMaxPooling1D())

    # Adding the dense hidden layer with ReLU activation
    model.add(keras.layers.Dense(hidden_dims, activation='relu'))

    # Adding the output layer with sigmoid activation for binary classification
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    loss_function = keras.losses.BinaryCrossentropy(from_logits=True)
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    # Compiling the model with binary cross-entropy loss and Adam optimizer
    model.compile(loss=loss_function, optimizer='adam', metrics=['accuracy'])

    # Training the model
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), callbacks=[callback])

    # Save model
    model.save('model/modelCNN.keras')

    # Predicting the probabilities for test data
    y_pred_prob = model.predict(x_test)

    # Converting the probabilities to binary classes based on threshold
    y_pred = (y_pred_prob > threshold).astype(int)

    # Calculating the evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Printing the evaluation metrics
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1-score:', f1)

    return accuracy, precision, recall, f1, history, y_pred

def TestTransformer(x_train, y_train, x_test, y_test):
    embed_dim = 32  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer
    vocab_size = 20000  # Only consider the top 20k words
    maxlen = 99  # Only consider the first 200 words of each movie review
    threshold = 0.5  # Threshold for binary classification

    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = transformer.TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = transformer.TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(1, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=32, epochs=2, validation_data=(x_test, y_test))

    # Predicting the probabilities for test data
    y_pred_prob = model.predict(x_test)

    # Converting the probabilities to binary classes based on threshold
    y_pred = (y_pred_prob > threshold).astype(int)

    print(y_test)
    print(len(y_pred))
    print("--------------------------------------")
    print(y_pred)
    print(len(y_pred))

    # Calculating the evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Printing the evaluation metrics
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1-score:', f1)

    return accuracy, precision, recall, f1

def CNNwithATT(x_train, y_train, x_test, y_test):
    threshold = 0.5  # Threshold for binary classification

    # Define early stopping callback
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='loss',  # Metric to monitor
        patience=10,         # Number of epochs to wait without improvement 
    )

    # Custom Attention Layer
    class AttentionLayer(layers.Layer):
        def __init__(self, **kwargs):
            super(AttentionLayer, self).__init__(**kwargs)

        def build(self, input_shape):
            # The attention layer will work on the sequence length (axis 1)
            self.attention_weights = self.add_weight(
                name="attention_weights",
                shape=(input_shape[-1], 1),  # Shape is (num_filters, 1)
                initializer="uniform",
                trainable=True
            )
            super(AttentionLayer, self).build(input_shape)

        def call(self, inputs):
            # Apply the attention weights across the sequence dimension
            attention_scores = tf.matmul(inputs, self.attention_weights)  # Shape: (batch_size, sequence_length, 1)
            attention_scores = tf.nn.softmax(attention_scores, axis=1)  # Normalize across the sequence length
            attention_scores = tf.squeeze(attention_scores, axis=-1)  # Shape: (batch_size, sequence_length)

            # Apply the attention scores to the inputs
            weighted_sum = inputs * attention_scores[..., tf.newaxis]  # Shape: (batch_size, sequence_length, num_filters)
            return tf.reduce_sum(weighted_sum, axis=1)  # Shape: (batch_size, num_filters)

    # Define CNN + Attention Model
    model = keras.Sequential([
        keras.layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=(x_train.shape[1], 1)),  # 66 features per sample
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Conv1D(128, kernel_size=3, activation='relu'),
        keras.layers.MaxPooling1D(pool_size=2),
        AttentionLayer(),  # Attention layer after convolution
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(y_train.shape[1], activation='softmax')  # Output layer with softmax
    ])

    loss_function = keras.losses.BinaryCrossentropy(from_logits=True)
    # Compile the model
    model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])

    # Train the model with early stopping
    history = model.fit(
        x_train,  # Add a channel dimension for CNN
        y_train,
        validation_data=(x_test, y_test),
        epochs=50,  # Set a high maximum number of epochs
        batch_size=32,
        callbacks=[early_stopping]  # Add the early stopping callback here
    )

    # Save model
    model.save('model/modelCNNwithATT.keras')

    # Predicting the probabilities for test data
    y_pred_prob = model.predict(x_test)

    # Converting the probabilities to binary classes based on threshold
    y_pred = (y_pred_prob > threshold).astype(int)

    # Calculating the evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Printing the evaluation metrics
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1-score:', f1)

    return accuracy, precision, recall, f1, history, y_pred

def VGG(x_train, y_train, x_test, y_test):
    threshold = 0.5  # Threshold for binary classification
    # Define early stopping callback
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',  # Metric to monitor
        patience=10,         # Number of epochs to wait without improvement
    )

    # Define 1D VGG-like Model for 3D input
    model = keras.Sequential([
        # Block 1: Convolution + Pooling
        keras.layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=(x_train.shape[1], 1)),  # 66 features per sample
        keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
        keras.layers.MaxPooling1D(pool_size=2),

        # Block 2: Convolution + Pooling
        keras.layers.Conv1D(128, kernel_size=3, activation='relu'),
        keras.layers.Conv1D(128, kernel_size=3, activation='relu'),
        keras.layers.MaxPooling1D(pool_size=2),

        # Block 3: Convolution + Pooling
        keras.layers.Conv1D(256, kernel_size=3, activation='relu'),
        keras.layers.Conv1D(256, kernel_size=3, activation='relu'),
        keras.layers.MaxPooling1D(pool_size=2),

        # Flatten the features before passing to the dense layers
        keras.layers.Flatten(),

        # Dense Layer
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.5),  # Dropout for regularization

        # Output Layer with softmax activation
        keras.layers.Dense(y_train.shape[1], activation='softmax')  # Output layer for multi-class classification
    ])

    loss_function = keras.losses.BinaryCrossentropy(from_logits=True)

    # Compile the model
    model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])

    # Train the model with early stopping
    history = model.fit(
        x_train,  # Add a channel dimension for CNN
        y_train,
        validation_data=(x_test, y_test),
        epochs=50,  # Set a high maximum number of epochs
        batch_size=32,
        callbacks=[early_stopping]  # Add the early stopping callback here
    )

    # Save model
    model.save('model/modelVGG.keras')

    # Predicting the probabilities for test data
    y_pred_prob = model.predict(x_test)

    # Converting the probabilities to binary classes based on threshold
    y_pred = (y_pred_prob > threshold).astype(int)

    # Calculating the evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Printing the evaluation metrics
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1-score:', f1)

    return accuracy, precision, recall, f1, history, y_pred