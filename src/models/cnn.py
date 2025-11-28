import tensorflow as tf


class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()

        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3)),
            tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3)),
            tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3)),
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
    
    def loss(self, predictions: tf.Tensor, targets: tf.Tensor) -> tf.Tensor:
        """
        Computes the cross-entropy loss between predicted and target action values.
        :predictions: tf.Tensor of shape (batch_size, 36) representing the predicted action values.
        :targets: tf.Tensor of shape (batch_size, 36) representing the target action values.
        :returns: tf.Tensor representing the computed loss.
        """
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        loss = loss_fn(targets, predictions)
        return loss