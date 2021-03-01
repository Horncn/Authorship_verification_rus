from tensorflow import keras
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split

def train_model(train_data, train_labels, verbose=1):
    train_part = 0.8
    x_train, x_test, y_train, y_test = train_test_split(train_data, train_labels, train_size=train_part)
    epochs = 100
    batch_size = 16
    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(2, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, verbose=verbose, batch_size=batch_size, epochs=epochs)

    test_loss, test_acc = model.evaluate(x_test,
                                         y_test,
                                         batch_size=1,
                                         verbose=verbose)
    model.evaluate(train_data[::3], train_labels[::3], batch_size=4)
    model.evaluate(train_data[1::3], train_labels[1::3], batch_size=4)
    model.save('model')
    if verbose:
        get_data(train_data, train_labels)
    return test_acc


def get_data(train_data, train_labels):
    _, x_test, _, y_test = train_test_split(train_data, train_labels, train_size=0.8)
    model = keras.models.load_model('model')
    y_test = y_test.argmax(axis=1)
    y_pred = model.predict(x_test, batch_size=2, verbose=1)
    y_pred = np.argmax(y_pred, axis=1)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
