import os
import tensorflow as tf
import numpy as np
from models.cnn import CNN
from encoding.full_encoding import encode_full, make_action_mask
from constants import BrickType
try:
    import wandb as _wandb
    WAND_AVAILABLE = True
except Exception:
    _wandb = None
    WAND_AVAILABLE = False


class Agent:
    def __init__(
        self,
        env,
        model,
        num_episodes=20,
        epochs=3,
        gamma=0.99,
        entropy_coef=0.01,
        temperature=1.0,
        value_coef=0.5,
    ):
        self.env = env
        self.model = model             
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

        self.num_episodes = num_episodes
        self.epochs = max(1, min(int(epochs), 3))
        self.batch_size = 64
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.temperature = temperature
        self.value_coef = value_coef
        self.state = encode_full(
            self.env.player_board,
            self.env.player_board != BrickType.UNKNOWN,
            channels_first=False,
        )
        self.value_net = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=self.state.shape),
                tf.keras.layers.Conv2D(
                    16, 3, padding="same", activation="relu"
                ),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(1), 
            ]
        )

        self.total_reward = 0.0
        self.games_played = 0
        self.current_episode = 0
        self.episodes = []

    def reset(self):
        """Reset the environment and return the initial encoded observation."""
        board, info = self.env.reset()
        self.state = encode_full(
            board,
            board != BrickType.UNKNOWN,
            channels_first=False,
        )
        self.total_reward = 0.0
        return self.state

    def _sample_action_and_logprob(self, state, revealed_mask):
        """
        Given raw state and revealed_mask, compute masked logits, sample action,
        and return (action, log_prob_of_action, encoded_state).
        """
        state_encoded = encode_full(
            state,
            revealed_mask,
            channels_first=False,
        )

        input_tensor = tf.convert_to_tensor(state_encoded, dtype=tf.float32)
        input_tensor = tf.expand_dims(input_tensor, axis=0) 

        logits = self.model(input_tensor, isTesting=False)
        logits = tf.reshape(logits, (-1,)) 
        logits = tf.clip_by_value(logits, -10.0, 10.0)
        action_mask = make_action_mask(revealed_mask, flatten=True)
        mask_tf = tf.convert_to_tensor(action_mask, dtype=tf.bool)

        if not tf.reduce_any(mask_tf):
            return None, None, state_encoded
        masked_logits = tf.where(
            mask_tf,
            logits,
            tf.fill(tf.shape(logits), -1e9),
        )
        scaled_logits = masked_logits / self.temperature
        action = tf.random.categorical(scaled_logits[None, :], 1)[0, 0]
        action = int(action.numpy())
        log_probs = tf.nn.log_softmax(scaled_logits)
        log_prob = log_probs[action]

        return action, log_prob, state_encoded

    def collect_episode(self):
        """
        Collect one full episode using current policy.
        Returns (episode, episode_reward).

        episode = list of transitions:
          {
            "state": encoded_state,
            "action": int,
            "reward": float,
            "done": bool
          }
        """
        episode = []
        state, info = self.env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            revealed_mask = (state != BrickType.UNKNOWN)

            action, log_prob, state_encoded = self._sample_action_and_logprob(
                state, revealed_mask
            )

            if action is None:
                break

            self.env.button = "left"
            next_state, reward, done, info = self.env.step(action)
            episode_reward += reward

            episode.append(
                {
                    "state": state_encoded,
                    "action": action,
                    "reward": float(reward),
                    "done": done,
                }
            )

            state = next_state

        self.current_episode += 1
        return episode, episode_reward

    def compute_returns(self, episode):
        """
        Discounted returns: G_t = r_t + gamma * G_{t+1}
        """
        returns = []
        G = 0.0
        for transition in reversed(episode):
            G = transition["reward"] + self.gamma * G
            returns.insert(0, G)
        return returns

    def play(self):
        """
        Collect num_episodes episodes and then train.

        Keeps the same interface your main() expects.
        """
        episode, episode_reward = self.collect_episode()
        self.episodes.append(episode)

        if len(self.episodes) >= self.num_episodes:
            self.train()
            self.episodes = []
            self.current_episode = 0

    def train(self):
        """
        Advantage Actor–Critic training step.

        - Actor: policy gradient with advantages = returns - V(s)
        - Critic: regression to (unnormalized) returns
        - Entropy regularization for exploration
        """
        if len(self.episodes) == 0:
            print("Warning: No episodes collected, skipping training")
            return

        all_states = []
        all_actions = []
        all_returns = []
        for ep in self.episodes:
            returns = self.compute_returns(ep)  
            for t, transition in enumerate(ep):
                all_states.append(transition["state"])
                all_actions.append(transition["action"])
                all_returns.append(returns[t])

        if len(all_states) == 0:
            print("Warning: No transitions collected, skipping training")
            return

        states = tf.convert_to_tensor(np.array(all_states), dtype=tf.float32)
        actions = tf.convert_to_tensor(np.array(all_actions), dtype=tf.int32)
        returns = tf.convert_to_tensor(np.array(all_returns), dtype=tf.float32)
        dataset = tf.data.Dataset.from_tensor_slices(
            {"states": states, "actions": actions, "returns": returns}
        )
        dataset = dataset.shuffle(buffer_size=len(all_states)).batch(self.batch_size)

        for epoch in range(self.epochs):
            epoch_loss = tf.keras.metrics.Mean()
            epoch_entropy = tf.keras.metrics.Mean()

            for batch in dataset:
                batch_states = batch["states"]
                batch_actions = batch["actions"]
                batch_returns = batch["returns"] 

                with tf.GradientTape() as tape:
                    logits = self.model(batch_states, isTesting=False)
                    logits = tf.reshape(logits, (tf.shape(logits)[0], -1))
                    logits = tf.clip_by_value(logits, -10.0, 10.0)
                    values = self.value_net(batch_states)   
                    values = tf.squeeze(values, axis=-1)  

                    unknown_channels = batch_states[:, :, :, 9]  
                    revealed_mask = unknown_channels < 0.5      
                    action_masks = ~revealed_mask               
                    action_masks = tf.reshape(
                        action_masks, (tf.shape(action_masks)[0], -1)
                    )

                    masked_logits = tf.where(
                        action_masks,
                        logits,
                        tf.fill(tf.shape(logits), -1e9),
                    )
                    scaled_logits = masked_logits / self.temperature

                    log_probs = tf.nn.log_softmax(scaled_logits)
                    probs = tf.nn.softmax(scaled_logits)
                    batch_size = tf.shape(batch_actions)[0]
                    indices = tf.stack([tf.range(batch_size), batch_actions], axis=1)
                    selected_log_probs = tf.gather_nd(log_probs, indices)
                    advantages = batch_returns - values
                    adv_mean = tf.reduce_mean(advantages)
                    adv_std = tf.math.reduce_std(advantages) + 1e-8
                    norm_advantages = (advantages - adv_mean) / adv_std
                    actor_loss = -tf.reduce_mean(selected_log_probs * norm_advantages)
                    critic_loss = tf.reduce_mean(tf.square(batch_returns - values))
                    entropy = -tf.reduce_sum(probs * log_probs, axis=1)
                    entropy_loss = -self.entropy_coef * tf.reduce_mean(entropy)

                    loss = actor_loss + self.value_coef * critic_loss + entropy_loss
                vars_all = (
                    self.model.trainable_variables
                    + self.value_net.trainable_variables
                )
                grads = tape.gradient(loss, vars_all)
                grads = [
                    tf.clip_by_norm(g, 1.0) if g is not None else None
                    for g in grads
                ]

                self.optimizer.apply_gradients(zip(grads, vars_all))

                epoch_loss.update_state(loss)
                epoch_entropy.update_state(tf.reduce_mean(entropy))

            loss_val = float(epoch_loss.result().numpy())
            entropy_val = float(epoch_entropy.result().numpy())
            print(
                f"  Epoch {epoch + 1}/{self.epochs}: Loss={loss_val:.4f}, Entropy={entropy_val:.4f}"
            )

            if WAND_AVAILABLE:
                try:
                    _wandb.log(
                        {
                            "train/epoch_loss": loss_val,
                            "train/entropy": entropy_val,
                            "epoch": epoch + 1,
                        }
                    )
                except Exception:
                    pass
        try:
            os.makedirs("models", exist_ok=True)
            ckpt_path = os.path.join("models", "minesweeper_rl_latest.h5")
            self.model.save(ckpt_path)
            if WAND_AVAILABLE:
                try:
                    _wandb.save(ckpt_path)
                except Exception:
                    pass
        except Exception:
            pass

        return loss


    def choose_action(self, board, deterministic=False):
        """
        Given the current player board, return an action using the policy.
        """
        revealed_mask = (board != BrickType.UNKNOWN)
        board_encoded = encode_full(
            board,
            revealed_mask,
            channels_first=False,
        )

        input_tensor = tf.convert_to_tensor(board_encoded, dtype=tf.float32)
        input_tensor = tf.expand_dims(input_tensor, axis=0)

        logits = self.model(input_tensor, isTesting=True)
        logits = tf.reshape(logits, (-1,))
        logits = tf.clip_by_value(logits, -10.0, 10.0)

        action_mask = make_action_mask(revealed_mask, flatten=True)
        mask_tf = tf.convert_to_tensor(action_mask, dtype=tf.bool)

        if not tf.reduce_any(mask_tf):
            return None

        masked_logits = tf.where(
            mask_tf,
            logits,
            tf.fill(tf.shape(logits), -1e9),
        )
        scaled_logits = masked_logits / self.temperature

        if deterministic:
            action = int(tf.argmax(scaled_logits).numpy())
        else:
            action = tf.random.categorical(scaled_logits[None, :], 1)[0, 0]
            action = int(action.numpy())

        return action

    def validate(self, num_games=1000):
        """
        Plays `num_games` complete Minesweeper games using the current policy.
        Returns win rate and average steps survived.
        """
        wins = 0
        total_steps = 0
        total_reward = 0.0

        for _ in range(num_games):
            state, info = self.env.reset()
            done = False
            steps = 0
            episode_reward = 0.0

            while not done:
                action = self.choose_action(state, deterministic=True)
                if action is None:
                    break

                self.env.button = "left"
                state, reward, done, info = self.env.step(action)
                steps += 1
                episode_reward += reward

            if info.get("status") == "win":
                wins += 1

            total_steps += steps
            total_reward += episode_reward

        win_rate = wins / num_games * 100.0
        avg_steps = total_steps / num_games
        avg_reward = total_reward / num_games

        print(f"\nVALIDATION — {num_games} Games")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Avg Steps Survived: {avg_steps:.2f}")
        print(f"Avg Episode Reward: {avg_reward:.2f}\n")

        if WAND_AVAILABLE:
            try:
                _wandb.log(
                    {
                        "eval/win_rate": win_rate,
                        "eval/avg_steps": avg_steps,
                        "eval/avg_reward": avg_reward,
                    }
                )
            except Exception:
                pass

        return win_rate, avg_steps
