import tensorflow as tf
import numpy as np
from src.models.cnn import CNN
from src.minesweeper.dataset import generate_examples
from src.encoding.full_encoding import encode_full, make_action_mask
from src.constants import BrickType

def convert_board(board):
    label = np.where(board == BrickType.MINE, 1.0, 0.0).reshape(-1)
    return label

class Agent:
    def __init__(self, env, model, num_samples = 320, epochs = 10):
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
        self.env = env
        self.num_samples = num_samples
        self.epochs = epochs
        self.batch_size = 32
        self.state = encode_full(self.env.player_board,
                                 self.env.player_board != BrickType.UNKNOWN,
                                 channels_first=False)
        
        self.total_reward = 0.0
        self.games_played = 0
        self.features = np.zeros((num_samples, self.env.num_rows, self.env.num_cols, self.state.shape[-1]))
        self.labels = np.zeros((num_samples, self.env.action_space.n))
        self.current_sample = 0


    def reset(self):
        """Reset the environment and return the initial observation."""
        board, info = self.env.reset()

        self.state = encode_full(board,
                             board != BrickType.UNKNOWN,
                             channels_first=False) 
        self.total_reward = 0.0
        return self.state

    def play(self):
        """Play one episode using the trained model."""
        done = False

        input_tensor = tf.convert_to_tensor(self.state, dtype=tf.float32)
        input_tensor = tf.expand_dims(input_tensor, axis=0)  
 
        output_tensor = self.model(input_tensor, isTesting=False)
        output_tensor = tf.reshape(output_tensor, (1, -1)) 

        
        mask = make_action_mask(self.state[:,:,-1] == 1, flatten=True)
        mask = tf.expand_dims(mask, axis=0)
      
        output_tensor = tf.where(mask, tf.fill(tf.shape(output_tensor), np.inf), output_tensor)
        min_index = tf.argmin(output_tensor[0]).numpy()

        valid_actions = output_tensor[0] != np.inf
        if not np.any(valid_actions):
            done = True
            
        else:
            action = int(min_index)  # min_index is already the best valid action
            self.env.button = 'left'
            next_state, reward, done, info = self.env.step(action)
            # self.total_reward += reward
            self.state = encode_full(next_state,
                                 next_state != BrickType.UNKNOWN,
                                 channels_first=False)
            self.add_to_dataset(self.state, convert_board(self.env.board))
        


    def train(self):
        """
        Train the CNN model using generated dataset examples.
        
        Returns:
            loss: The final loss value after training."""

        for epoch in range(self.epochs):
            # X, M, Y, V = generate_examples(n_examples=self.num_samples)
            features = tf.convert_to_tensor(self.features, dtype=tf.float32)
            labels = tf.convert_to_tensor(self.labels, dtype=tf.float32)
            dataset = tf.data.Dataset.from_tensor_slices((features, labels))
            dataset = dataset.shuffle(buffer_size=self.num_samples).batch(self.batch_size)
            epoch_loss = tf.keras.metrics.Mean()
            epoch_accuracy = tf.keras.metrics.BinaryAccuracy() 

            for batch_features, batch_labels in dataset:
                with tf.GradientTape() as tape:
                    predictions = self.model(batch_features, isTesting=False)
                    loss = self.model.loss(predictions, batch_labels)
                 
                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                epoch_loss.update_state(loss)
                epoch_accuracy.update_state(batch_labels, predictions)
           
            print(f"Epoch {epoch + 1}, Loss: {epoch_loss.result().numpy():.4f}, Accuracy: {epoch_accuracy.result().numpy():.4f}")

        return loss
    
    def add_to_dataset(self, feature, label):
        self.features[self.current_sample] = feature
        self.labels[self.current_sample] = label
        self.current_sample += 1
        if self.current_sample >= self.num_samples:
            self.train()
            self.current_sample = 0
    