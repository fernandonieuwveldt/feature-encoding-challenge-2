
"""
workflow for training and predicting
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas
import tensorflow as tf
from sklearn.model_selection import train_test_split

from encoding_challenge.src.data import TFDataTransformer
from encoding_challenge.src.feature_encoder import CategoricalFeatureEncoder, EmbeddingFeatureEncoder


raw_data = pandas.read_csv("encoding_challenge/data/train.csv")
train_data_features = raw_data.drop(['id', 'target'], axis=1)
train_data_labels = raw_data.pop('target')
test_data_features = pandas.read_csv("encoding_challenge/data/test.csv")

train_data_features.fillna('missing', inplace=True)
test_data_features.fillna('missing', inplace=True)
train_data_features = train_data_features.astype('str')
test_data_features = test_data_features.astype('str')

CATEGORICAL_FEATURES = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4',
                        'nom_0', 'ord_0', 'nom_4', 'ord_1', 'nom_3',
                        'nom_1', 'nom_2', 'ord_2', 'day', 'month',
                        'ord_3', 'ord_4']

EMBEDDING_FEATURES = ['ord_5', 'nom_7', 'nom_8', 'nom_5', 'nom_6', 'nom_9']

X_train, X_test, y_train, y_test = train_test_split(train_data_features, train_data_labels)

BATCH_SIZE = 1024
train_dataset = TFDataTransformer().transform(X_train, y_train).batch(BATCH_SIZE)
val_dataset = TFDataTransformer().transform(X_test, y_test).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(dict(test_data_features.drop('id', axis=1))).batch(BATCH_SIZE)

categorical_inputs, categorical_feature_encoders = CategoricalFeatureEncoder(CATEGORICAL_FEATURES).encode(X_train)
embedding_inputs, embedding_feature_encoders = EmbeddingFeatureEncoder(EMBEDDING_FEATURES).encode(X_train)

feature_layer_inputs = {**categorical_inputs, **embedding_inputs}
feature_columns_wide = [feature_encoded for _, feature_encoded in categorical_feature_encoders.items()]
feature_columns_deep = [feature_encoded for _, feature_encoded in embedding_feature_encoders.items()]


def wide_and_deep_classifier(inputs, linear_feature_columns, dnn_feature_columns, dnn_hidden_units):
    """
    Setup a wide and deep model with Keras
    """
    metrics = [tf.keras.metrics.BinaryAccuracy(name='accuracy'),
               tf.keras.metrics.AUC(name='auc')]
    deep = tf.keras.layers.DenseFeatures(dnn_feature_columns)(inputs)

    for numnodes in dnn_hidden_units:
        deep = tf.keras.layers.Dense(numnodes, activation='relu')(deep)
        deep = tf.keras.layers.Dropout(0.5)(deep)
    wide = tf.keras.layers.DenseFeatures(linear_feature_columns)(inputs)
    combined = tf.keras.layers.concatenate([deep, wide], axis=1)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(combined)
    model = tf.keras.Model(inputs=[v for v in inputs.values()], outputs=output)
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(lr=0.001),
        metrics=metrics)
    return model


model = wide_and_deep_classifier(feature_layer_inputs, feature_columns_wide, feature_columns_deep, [256, 64, 64])

early_stopping = tf.keras.callbacks.EarlyStopping(**{'monitor': 'val_loss',
                                                     'mode': 'min',
                                                     'verbose': 1,
                                                     'patience': 10})

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(**{'filepath': '/tmp/best_model',
                                                         'monitor': 'val_loss',
                                                         'mode': 'min',
                                                         'verbose': 1,
                                                         'save_weights_only': True,
                                                         'save_best_only': True})

train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

model.fit(train_dataset,
          epochs=100,
          validation_data=val_dataset,
          callbacks=[early_stopping, model_checkpoint])

model.load_weights('/tmp/best_model')
preds = model.predict(test_dataset)


def submit_run(file_name=None):
    submit_data_frame = pandas.DataFrame([], columns=['id', 'target'])
    submit_data_frame['id'] = test_data_features['id'].values
    submit_data_frame['target'] = preds
    submit_data_frame.to_csv(f'{file_name}.csv', index=False)


submit_run('results')
