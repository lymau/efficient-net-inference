from keras import backend as K
import os
from keras.models import load_model

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def initialize_model(model_path: str):
    metrics = {
        "accuracy": "accuracy",
        "f1_m": f1_m,
        "precision_m": precision_m,
        "recall_m": recall_m,
    }
    return load_model(
        os.path.join(CURRENT_PATH, model_path), custom_objects=metrics, compile=True
    )
