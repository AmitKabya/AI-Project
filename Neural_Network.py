import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import PCA
from Data import Data, MissingData


def read_data(dr, md):
    if not md:
        data = Data(dr)
    else:
        data = MissingData(dr)
    return data.dataF, data


def df_to_dataset(dataframe, batch_size, shuffle=True):
    dataframe = dataframe.copy()
    labels = dataframe.pop('classes')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


def train_model(dataframe, batch_size, epochs):
    train, test = train_test_split(dataframe, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)

    feature_columns = []
    headers = list(dataframe.columns)
    headers.pop(0)

    for header in headers:
        feature_columns.append(feature_column.numeric_column(header))

    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

    train_ds = df_to_dataset(train, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

    model = tf.keras.Sequential([
        feature_layer,
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dropout(.1),
        layers.Dense(1)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    loss, accuracy = model.evaluate(test_ds)
    print("Accuracy", accuracy)
    return history, model, accuracy


def missing_data_prediction(data, model, batch_size):
    ds = df_to_dataset(data.missingDataF, batch_size=batch_size, shuffle=False)
    predictions = model.predict(ds, batch_size)

    y_predicted = []
    for i in range(len(predictions)):
        y_predicted.append(np.argmax(predictions[i]))

    missingMushrooms = data.missingDataF.copy()
    missingMushrooms['classes'] = y_predicted
    print(f"class prediction:\n{missingMushrooms.head(len(missingMushrooms))}")
    draw_results(data.missingDataF, y_predicted)


def draw_results(X_test, y_predicted):
    # test visualization:
    pca = PCA.pca_predict(X_test, y_predicted, 3)
    fig = plt.figure()
    fig.suptitle("Test Visualization")
    PCA.plot_pca(pca, fig)
    plt.show()


def draw_loss(history, epochs):
    loss_train = history.history['loss']
    loss_val = history.history['val_loss']
    epochs = range(1, epochs + 1)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(epochs, loss_train, 'g', label='Training loss')
    ax.plot(epochs, loss_val, 'b', label='validation loss')
    ax.title.set_text('Training and Validation loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid()


def draw_accuracy(history, epochs):
    loss_train = history.history['accuracy']
    loss_val = history.history['val_accuracy']
    epochs = range(1, epochs + 1)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(epochs, loss_train, 'g', label='Training accuracy')
    ax.plot(epochs, loss_val, 'b', label='validation accuracy')
    ax.title.set_text('Training and Validation accuracy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid()


def main(dr=0, md=False, batch_size=32, epochs=30):
    if not md:
        df = read_data(dr, md)[0]
        history, _, acc = train_model(df, batch_size=batch_size, epochs=epochs)
        draw_loss(history, epochs=epochs)
        draw_accuracy(history, epochs=epochs)
        plt.show()
    else:
        df, data = read_data(dr, md)
        history, model, acc = train_model(df, batch_size=batch_size, epochs=epochs)
        draw_loss(history, epochs=epochs)
        draw_accuracy(history, epochs=epochs)
        plt.show()
        missing_data_prediction(data, model, batch_size)

    return acc
