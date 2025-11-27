import tensorflow as tf
import numpy as np
from src.models.cnn import CNN
from src.minesweeper.dataset import generate_examples


class Agent:
    def __init__(self, env, model, num_samples = 320, epochs = 10):
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
        self.env = env
        self.num_samples = num_samples
        self.epochs = epochs
        self.batch_size = 32
        self.state = env.reset()


    def train(self):
        """
        Train the CNN model using generated dataset examples.
        
        Returns:
            loss: The final loss value after training."""
        print("Entered training function")

        for epoch in range(self.epochs):
            X, M, Y, V = generate_examples(n_examples=self.num_samples)
            features = tf.convert_to_tensor(X, dtype=tf.float32)
            labels = tf.convert_to_tensor(Y, dtype=tf.float32)
            dataset = tf.data.Dataset.from_tensor_slices((features, labels))
            dataset = dataset.shuffle(buffer_size=self.num_samples).batch(self.batch_size)
            epoch_loss = tf.keras.metrics.Mean()

            for batch_features, batch_labels in dataset:
                with tf.GradientTape() as tape:
                    predictions = self.model(batch_features, isTesting=False)
                    loss = self.model.loss(predictions, batch_labels)
                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                epoch_loss.update_state(loss)
           
            print(f"Epoch {epoch + 1}, Loss: {epoch_loss.result().numpy():.4f}")

        return loss
    