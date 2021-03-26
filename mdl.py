from tensorflow import keras
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_model(train_data, train_labels, verbose=1):
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])
    train_data = num_pipeline.fit_transform(train_data)
    train_part = 0.8
    x_train, x_test, y_train, y_test = train_test_split(train_data, train_labels, train_size=train_part,
                                                        random_state=42)
    x_train = num_pipeline.fit_transform(x_train)
    epochs = 100
    batch_size = epochs

    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, verbose=verbose, batch_size=batch_size, epochs=epochs)
    model.save('model')
    if verbose:
        get_data(x_test, y_test)
    # return test_acc


def f_05_u_score(true_y, pred_y, pos_label=1, threshold=0.5):
    """
    Return F0.5u score of prediction.
    :param true_y: true labels
    :param pred_y: predicted labels
    :param threshold: indication for non-decisions (default = 0.5)
    :param pos_label: positive class label (default = 1)
    :return: F0.5u score
    """

    n_tp = 0
    n_fn = 0
    n_fp = 0
    n_u = 0

    for i, pred in enumerate(pred_y):
        if pred == threshold:
            n_u += 1
        elif pred == pos_label and pred == true_y[i]:
            n_tp += 1
        elif pred == pos_label and pred != true_y[i]:
            n_fp += 1
        elif true_y[i] == pos_label and pred != true_y[i]:
            n_fn += 1

    return (1.25 * n_tp) / (1.25 * n_tp + 0.25 * (n_fn + n_u) + n_fp)


def get_data(x_test, y_test):
    model = keras.models.load_model('model')
    y_test = y_test.argmax(axis=1)
    y_pred = model.predict(x_test, batch_size=2, verbose=1)
    y_pred = np.argmax(y_pred, axis=1)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print('AUC ' + str(roc_auc_score(y_test, y_pred)))
    print('f05 ' + str(f_05_u_score(y_test, y_pred, pos_label=0)))
