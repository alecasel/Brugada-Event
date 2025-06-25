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


class CrossLeadAttention(layers.Layer):
    """Attention mechanism across ECG leads"""

    def __init__(self, num_heads=4, key_dim=32, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim
        )

    def call(self, inputs):
        # inputs: [batch, num_leads, features]
        attended = self.mha(inputs, inputs)
        return attended


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


def build_lead_feature_extractor():
    """CNN feature extractor for individual ECG leads"""
    return tf.keras.Sequential([
        layers.Conv1D(64, kernel_size=3, activation='relu',
                      kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Conv1D(64, kernel_size=3, activation='relu',
                      kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        layers.Conv1D(32, kernel_size=3, activation='relu',
                      kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Conv1D(32, kernel_size=3, activation='relu',
                      kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3)
    ])


def build_risk_stratification_model(seq_length,
                                    num_leads=12,
                                    unified_approach=True,
                                    risk_output_type='probability'):
    """
    Build model for Brugada syndrome risk stratification

    Args:
        seq_length: Length of ECG sequence
        num_leads: Number of ECG leads (default 12)
        risk_output_type: 'probability' (0-1) or 'classification'
        (low/high risk)
    """

    # Input per tutte le derivazioni
    inputs = Input(shape=(seq_length, num_leads))

    # Preprocessing
    x = layers.BatchNormalization()(inputs)
    x = layers.GaussianNoise(0.1)(x)  # Ridotto per multi-lead
    x = layers.BatchNormalization()(x)

    # Opzione 1: CNN su tutte le derivazioni insieme
    # (più semplice ma meno flessibile)
    if unified_approach:  # Approccio unified
        x = layers.Conv1D(128, kernel_size=5, activation='relu',
                          kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(128, kernel_size=5, activation='relu',
                          kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(0.4)(x)

        x = layers.Conv1D(64, kernel_size=3, activation='relu',
                          kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(64, kernel_size=3, activation='relu',
                          kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(0.4)(x)

        x = layers.Conv1D(32, kernel_size=3, activation='relu',
                          kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(32, kernel_size=3, activation='relu',
                          kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(0.4)(x)

    else:  # Approccio per-lead (più complesso)
        # Custom layer to extract individual leads
        class LeadExtractor(layers.Layer):
            def __init__(self, lead_index, **kwargs):
                super(LeadExtractor, self).__init__(**kwargs)
                self.lead_index = lead_index

            def call(self, inputs):
                # Extract single lead and add channel dimension
                return tf.expand_dims(inputs[:, :, self.lead_index], axis=-1)

            def get_config(self):
                config = super().get_config()
                config.update({"lead_index": self.lead_index})
                return config

        # Custom layer to stack lead features
        class LeadFeatureStacker(layers.Layer):
            def call(self, inputs):
                # inputs is a list of lead features
                return tf.stack(inputs, axis=1)

        # Estrai features da ogni derivazione separatamente
        feature_extractor = build_lead_feature_extractor()
        lead_features = []

        for i in range(num_leads):
            # Extract single lead using custom layer
            lead_input = LeadExtractor(i)(inputs)
            lead_feature = feature_extractor(lead_input)
            # Global average pooling per ogni derivazione
            lead_feature = layers.GlobalAveragePooling1D()(lead_feature)
            lead_features.append(lead_feature)

        # Stack features using custom layer: [batch, num_leads, features]
        x = LeadFeatureStacker()(lead_features)

        # Cross-lead attention
        x = CrossLeadAttention(num_heads=4, key_dim=32)(x)

        # Flatten per processing successivo
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.4)(x)

    # Temporal modeling con LSTM
    if len(x.shape) > 2:  # Se abbiamo ancora dimensione temporale
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)

        # Attention temporale
        attention_output = AttentionLayer()(x)
        x = attention_output

    # Risk prediction head
    x = layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(32, activation='relu',
                     kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    # Output layer
    if risk_output_type == 'probability':
        # Probabilità continua di rischio (0-1)
        outputs = layers.Dense(1, activation='sigmoid',
                               name='risk_probability')(x)
    elif risk_output_type == 'classification':
        # Classificazione basso/alto rischio
        outputs = layers.Dense(2, activation='softmax', name='risk_class')(x)
    else:
        raise ValueError(
            "risk_output_type must be 'probability' or 'classification'")

    model = models.Model(inputs=inputs, outputs=outputs)

    return model


def compile_risk_model(model):
    """Compile model with appropriate loss function"""

    from tensorflow.keras.optimizers import Adam

    optimizer = Adam(
        learning_rate=1e-4,
        beta_1=0.9,
        beta_2=0.99,
        epsilon=1e-7,
        amsgrad=True,
        weight_decay=1e-5,
        clipnorm=1.0,
    )

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'binary_accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    return model
