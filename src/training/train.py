import os
import tensorflow as tf
import numpy as np
from models.cnn import CNN
from minesweeper.dataset import generate_examples
from encoding.full_encoding import encode_full, make_action_mask
from constants import BrickType

# Optional Weights & Biases integration (graceful fallback if not installed)
try:
    import wandb as _wandb
    WAND_AVAILABLE = True
except Exception:
    _wandb = None
    WAND_AVAILABLE = False


def convert_board(board):
    label = np.where(board == BrickType.MINE, 1.0, 0.0).reshape(-1)
    return label


class Agent:
    def __init__(self, env, model, num_samples = 640, epochs = 10):
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.env = env
        self.num_samples = num_samples
        self.epochs = epochs
        self.batch_size = 64
        self.state = encode_full(self.env.player_board,
                                 self.env.player_board != BrickType.UNKNOWN,
                                 channels_first=False)
        
        self.total_reward = 0.0
        self.games_played = 0
        self.current_sample = 0

        self.features = np.zeros((num_samples, self.env.num_rows, self.env.num_cols, self.state.shape[-1]))
        self.labels = np.zeros((num_samples, self.env.action_space.n))
        self.masks = np.zeros((num_samples, self.env.action_space.n))


    def reset(self):
        """Reset the environment and return the initial observation."""
        board, info = self.env.reset()

        self.state = encode_full(board,
                             board != BrickType.UNKNOWN,
                             channels_first=False) 
        self.total_reward = 0.0
        return self.state


    def play(self):
        """Play one move using the CNN and store a supervised sample."""

        # Get the before state of the game
        before_board = self.env.player_board.copy()
        revealed_mask = (before_board != BrickType.UNKNOWN)

        # Encode the initial state
        before_encoded = encode_full(
            before_board,
            revealed_mask,
            channels_first=False
        )

        # Predict mine probabilities
        input_tensor = tf.convert_to_tensor(before_encoded, dtype=tf.float32)
        input_tensor = tf.expand_dims(input_tensor, axis=0) # (1, H, W, C)

        predictions = self.model(input_tensor, isTesting=False)
        predictions = tf.reshape(predictions, (1, -1))

        # Create action mask
        action_mask = make_action_mask(revealed_mask, flatten=True)
        mask_tf = tf.convert_to_tensor(action_mask, dtype=tf.bool)

        # Mask out revealed cells in predictions so they are not chosen
        masked_preds = tf.where(
            mask_tf[None, :],  # Add batch dimension
            predictions, 
            tf.fill(tf.shape(predictions), np.inf)
        )

        # Check if there are any valid actions left
        if not tf.reduce_any(mask_tf):
            # No valid actions left, reset environment
            _, info = self.env.reset()
            self.state = encode_full(
                self.env.player_board,
                self.env.player_board != BrickType.UNKNOWN,
                channels_first=False
            )
            return 

        # Pick the best valid action
        action = int(tf.argmin(masked_preds[0]).numpy())

        # Take the action
        self.env.button = 'left'
        next_state, reward, done, info = self.env.step(action)

        # Store the sample
        mine_map = convert_board(self.env.board)
        self.add_to_dataset(before_encoded, mine_map, action_mask)

        # Update agent state
        self.state = encode_full(
            next_state,
            next_state != BrickType.UNKNOWN,
            channels_first=False
        )

        if done:
            # Game over, reset environment
            _, info = self.env.reset()
            self.state = encode_full(
                self.env.player_board,
                self.env.player_board != BrickType.UNKNOWN,
                channels_first=False
            )
        

    def train(self):
        """
        Train the CNN model using generated dataset examples.
        
        Returns:
            loss: The final loss value after training."""

        features = tf.convert_to_tensor(self.features, dtype=tf.float32)
        labels = tf.convert_to_tensor(self.labels, dtype=tf.float32)
        masks = tf.convert_to_tensor(self.masks, dtype=tf.float32)

        dataset = tf.data.Dataset.from_tensor_slices({
            "features": features, 
            "labels": labels, 
            "masks": masks
        })
        dataset = dataset.shuffle(buffer_size=self.num_samples).batch(self.batch_size)
            
        for epoch in range(self.epochs):
            epoch_loss = tf.keras.metrics.Mean()

            for batch in dataset:
                batch_features = batch["features"]   # (32,6,6,10)
                batch_labels   = batch["labels"]     # (32,36)
                batch_masks    = batch["masks"]      # (32,36)
                
                with tf.GradientTape() as tape:
                    predictions = self.model(batch_features, isTesting=False)
                    loss = self.model.masked_bce_loss(predictions, batch_labels, batch_masks)
                 
                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                epoch_loss.update_state(loss)

            if epoch % 2 == 0 or epoch == self.epochs - 1:
                val = float(epoch_loss.result().numpy())
                print(f"Epoch {epoch + 1}, Loss: {val:.4f}")
                if WAND_AVAILABLE:
                    try:
                        _wandb.log({"train/epoch_loss": val, "epoch": epoch + 1})
                    except Exception:
                        pass

        # save a checkpoint after training round
        try:
            os.makedirs("models", exist_ok=True)
            ckpt_path = os.path.join("models", "minesweeper_cnn_latest.h5")
            self.model.save(ckpt_path)
            if WAND_AVAILABLE:
                try:
                    _wandb.save(ckpt_path)
                except Exception:
                    pass
        except Exception:
            pass

        return loss
    

    def add_to_dataset(self, feature, label, mask):
        # Add a new sample to the dataset
        self.features[self.current_sample] = feature
        self.labels[self.current_sample] = label
        self.masks[self.current_sample] = mask

        # Increment sample count, and train if full
        self.current_sample += 1
        if self.current_sample >= self.num_samples:
            self.train()
            self.current_sample = 0


    def choose_action(self, board):
        """
        Given the current player board, return the best action index
        using CNN predictions and your masking logic.
        """

        # Get revealed mask for encoding
        revealed_mask = (board != BrickType.UNKNOWN)

        # Encode the initial state
        board_encoded = encode_full(
            board,
            revealed_mask,
            channels_first=False
        )

        # Predict mine probabilities
        input_tensor = tf.convert_to_tensor(board_encoded, dtype=tf.float32)
        input_tensor = tf.expand_dims(input_tensor, axis=0) # (1, H, W, C)

        predictions = self.model(input_tensor, isTesting=True)
        predictions = tf.reshape(predictions, (1, -1))

        # Create action mask
        action_mask = make_action_mask(revealed_mask, flatten=True)
        mask_tf = tf.convert_to_tensor(action_mask, dtype=tf.bool)

        # Mask out revealed cells in predictions so they are not chosen
        masked_preds = tf.where(
            mask_tf[None, :],  # Add batch dimension
            predictions, 
            tf.fill(tf.shape(predictions), np.inf)
        )

        # Check if there are any valid actions left
        if not tf.reduce_any(mask_tf):
            return None

        # Pick the best valid action
        action = int(tf.argmin(masked_preds[0]).numpy())
        return action


    def validate(self, num_games=100):
        """
        Plays `num_games` complete Minesweeper games using the current CNN.
        Returns win rate and average steps survived.
        """

        wins = 0
        total_steps = 0

        for _ in range(num_games):
            # Reset environment
            state, info = self.env.reset()
            done = False
            steps = 0

            while not done:
                action = self.choose_action(state)
                
                if action is None:
                    break

                self.env.button = 'left'
                state, reward, done, info = self.env.step(action)
                steps += 1

            if info['status'] == 'win':
                wins += 1

            total_steps += steps

        win_rate = wins / num_games * 100.0
        avg_steps = total_steps / num_games

        print(f"\nVALIDATION — {num_games} Games")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Avg Steps Survived: {avg_steps:.2f}\n")

        if WAND_AVAILABLE:
            try:
                _wandb.log({"eval/win_rate": win_rate, "eval/avg_steps": avg_steps})
            except Exception:
                pass

        return win_rate, avg_steps

        