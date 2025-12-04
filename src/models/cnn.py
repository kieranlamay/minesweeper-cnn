import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy

class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()

        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), padding="same", activation='relu'),
            tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), padding="same", activation='relu'),
            tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), padding="same", activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units = 256, activation = 'relu'),
            tf.keras.layers.Dense(units = 256, activation = 'relu'),
            tf.keras.layers.Dense(units = 36)
        ])

    def call(self, inputs: tf.Tensor, isTesting: bool = False) -> tf.Tensor:
        """
        Runs a forward pass on an input tensor of the one-hot encoding minefield state.
        :inputs: tf.Tensor of shape (1, 6, 6, 10) representing the one-hot encoded minefield state.
        :isTesting: bool indicating whether the model is being run in testing mode.
        :returns: tf.Tensor of shape (1, 36) representing the output action values.
        """

        outputs = self.model(inputs)
        return outputs

    def masked_bce_loss(self, predictions: tf.Tensor, targets: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """
        Masked binary cross-entropy loss.
        
        predictions: (batch, H*W)
        targets:     (batch, H*W)
        mask:        (batch, H*W) with 1 for valid (unrevealed) tiles, 0 for ignored tiles
        """

        # BCE per element
        bce_fn = BinaryCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE
        )

        predictions = tf.expand_dims(predictions, axis=-1)   # (batch, 36, 1)
        targets     = tf.expand_dims(targets, axis=-1)       # (batch, 36, 1)

        bce = bce_fn(targets, predictions)

        # Convert mask to float
        mask = tf.cast(mask, tf.float32)

        # Apply mask
        masked_bce = bce * mask

        # Normalize by number of valid cells
        loss = tf.reduce_sum(masked_bce) / tf.reduce_sum(mask)

        return loss
    
    import tensorflow as tf

    def masked_weighted_bce_loss(self, logits, labels, mask, pos_weight=None, eps=1e-8):
        # logits, labels, mask: tensors shape (B, H*W)
        labels = tf.cast(labels, tf.float32)
        mask = tf.cast(mask, tf.float32)

        if pos_weight is None:
            pos = tf.reduce_sum(labels * mask)
            neg = tf.reduce_sum((1.0 - labels) * mask)
            pos_weight = tf.where(pos > 0, neg / (pos + eps), 1.0)

        per_entry = tf.nn.weighted_cross_entropy_with_logits(labels=labels, logits=logits, pos_weight=pos_weight)
        per_entry = per_entry * mask
        normalizer = tf.reduce_sum(mask) + eps
        return tf.reduce_sum(per_entry) / normalizer
