import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, Input


class AttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], input_shape[-1]),
            initializer="glorot_uniform",
            trainable=True
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[-1],),
            initializer="zeros",
            trainable=True
        )
        self.u = self.add_weight(
            name="context_vector",
            shape=(input_shape[-1],),
            initializer="glorot_uniform",
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        score = tf.tensordot(inputs, self.W, axes=1) + self.b
        score = tf.tanh(score)
        attention_weights = tf.nn.softmax(
            tf.tensordot(score, self.u, axes=1), axis=1)
        weighted_input = inputs * tf.expand_dims(attention_weights, -1)
        output = tf.reduce_sum(weighted_input, axis=1)
        return output


def build_supervised_model(seq_length,
                           num_features):

    inputs = Input(shape=(seq_length, num_features))

    x = layers.BatchNormalization()(inputs)
    x = layers.GaussianNoise(0.15)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(128, kernel_size=3, activation='relu',
                      kernel_regularizer=regularizers.l2(0.02))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, kernel_size=3, activation='relu',
                      kernel_regularizer=regularizers.l2(0.02))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv1D(64, kernel_size=3, activation='relu',
                      kernel_regularizer=regularizers.l2(0.02))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, kernel_size=3, activation='relu',
                      kernel_regularizer=regularizers.l2(0.02))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv1D(32, kernel_size=3, activation='relu',
                      kernel_regularizer=regularizers.l2(0.02))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(32, kernel_size=3, activation='relu',
                      kernel_regularizer=regularizers.l2(0.02))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    # Livello di Attenzione
    attention_output = AttentionLayer()(x)
    x = layers.Dense(32, activation='relu',
                     kernel_regularizer=regularizers.l2(0.02))(
                         attention_output)
    x = layers.Dense(32, activation='relu',
                     kernel_regularizer=regularizers.l2(0.02))(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(32, activation='relu',
                     kernel_regularizer=regularizers.l2(0.02))(x)
    x = layers.Dropout(0.5)(x)

    # NOTA: Ora usiamo 3 neuroni in output con attivazione softmax
    outputs = layers.Dense(3, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    return model


def create_student_model(seq_length,
                         num_features):
    """
    Student model for semi-supervised Brugada ECG classification.
    Outputs both classification logits and contrastive embeddings.
    """

    inputs = Input(shape=(seq_length, num_features))

    x = layers.BatchNormalization()(inputs)
    x = layers.GaussianNoise(0.15)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(128, kernel_size=3, activation='relu',
                      kernel_regularizer=regularizers.l2(0.02))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, kernel_size=3, activation='relu',
                      kernel_regularizer=regularizers.l2(0.02))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv1D(64, kernel_size=3, activation='relu',
                      kernel_regularizer=regularizers.l2(0.02))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, kernel_size=3, activation='relu',
                      kernel_regularizer=regularizers.l2(0.02))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv1D(32, kernel_size=3, activation='relu',
                      kernel_regularizer=regularizers.l2(0.02))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(32, kernel_size=3, activation='relu',
                      kernel_regularizer=regularizers.l2(0.02))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    # Attention layer
    attention_output = AttentionLayer()(x)

    x = layers.Dense(32, activation='relu',
                     kernel_regularizer=regularizers.l2(0.02))(
                         attention_output)
    x = layers.Dense(32, activation='relu',
                     kernel_regularizer=regularizers.l2(0.02))(x)
    x = layers.Dropout(0.5)(x)

    # Shared latent representation
    shared_repr = layers.Dense(32, activation='relu',
                               kernel_regularizer=regularizers.l2(0.02))(x)
    x = layers.Dropout(0.5)(shared_repr)

    # Classification head
    class_output = layers.Dense(3, activation='softmax', name='classifier')(x)

    # Contrastive head (embedding projection)
    contrastive_output = layers.Dense(
        32, activation=None, name='embedding')(shared_repr)
    contrastive_output = layers.Lambda(
        lambda y: tf.math.l2_normalize(y, axis=1))(contrastive_output)

    model = models.Model(inputs=inputs, outputs=[
        class_output, contrastive_output])

    return model
