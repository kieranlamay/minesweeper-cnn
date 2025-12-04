import os
import tensorflow as tf
import numpy as np
from models.cnn import CNN
from encoding.full_encoding import encode_full, make_action_mask
from constants import BrickType

# Optional Weights & Biases integration (graceful fallback if not installed)
try:
    import wandb as _wandb
    WAND_AVAILABLE = True
except Exception:
    _wandb = None
    WAND_AVAILABLE = False


class Agent:
    def __init__(self, env, model, num_episodes=20, epochs=10, gamma=0.99, entropy_coef=0.01):
        """
        RL Agent using REINFORCE (policy gradient) algorithm.
        
        Args:
            env: Minesweeper environment
            model: CNN model (outputs action logits)
            num_episodes: Number of episodes to collect before training
            epochs: Number of training epochs per batch
            gamma: Discount factor for future rewards
            entropy_coef: Entropy bonus coefficient for exploration
        """
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.env = env
        self.num_episodes = num_episodes
        self.epochs = epochs
        self.batch_size = 64
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.state = encode_full(self.env.player_board,
                                 self.env.player_board != BrickType.UNKNOWN,
                                 channels_first=False)
        
        self.total_reward = 0.0
        self.games_played = 0
        self.current_episode = 0
        
        # Storage for RL transitions: (state, action, reward, log_prob, done)
        self.episodes = []


    def reset(self):
        """Reset the environment and return the initial observation."""
        board, info = self.env.reset()

        self.state = encode_full(board,
                             board != BrickType.UNKNOWN,
                             channels_first=False) 
        self.total_reward = 0.0
        return self.state


    def collect_episode(self):
        """
        Collect one full episode using the current policy.
        Returns the episode as a list of (state, action, reward, log_prob, done) tuples.
        """
        episode = []
        state, info = self.env.reset()
        done = False
        episode_reward = 0.0
        
        while not done:
            # Get revealed mask for encoding
            revealed_mask = (state != BrickType.UNKNOWN)
            
            # Encode the current state
            state_encoded = encode_full(
                state,
                revealed_mask,
                channels_first=False
            )
            
            # Get policy logits
            input_tensor = tf.convert_to_tensor(state_encoded, dtype=tf.float32)
            input_tensor = tf.expand_dims(input_tensor, axis=0)  # (1, H, W, C)
            
            logits = self.model(input_tensor, isTesting=False)
            logits = tf.reshape(logits, (1, -1))  # (1, action_space)
            
            # Create action mask
            action_mask = make_action_mask(revealed_mask, flatten=True)
            mask_tf = tf.convert_to_tensor(action_mask, dtype=tf.bool)
            
            # Mask out invalid actions (set to large positive value so they get low probability)
            # Note: model outputs lower values for safer actions, so we negate for softmax
            masked_logits = tf.where(
                mask_tf[None, :],
                logits,
                tf.fill(tf.shape(logits), 1e9)  # Large positive value (will get low prob after negation)
            )
            
            # Check if there are any valid actions
            if not tf.reduce_any(mask_tf):
                break
            
            # Convert safety scores to probabilities: lower logits (safer) -> higher probability
            # Negate logits so lower values become higher probabilities after softmax
            negated_logits = -masked_logits[0]
            
            # Sample action from policy distribution
            probs = tf.nn.softmax(negated_logits)
            action = tf.random.categorical(tf.expand_dims(tf.math.log(probs + 1e-8), 0), 1)[0, 0]
            action = int(action.numpy())
            
            # Get log probability of selected action (using negated logits)
            log_probs = tf.nn.log_softmax(negated_logits)
            log_prob = log_probs[action]
            
            # Take the action
            self.env.button = 'left'
            next_state, reward, done, info = self.env.step(action)
            episode_reward += reward
            
            # Store transition
            episode.append({
                'state': state_encoded,
                'action': action,
                'reward': float(reward),
                'log_prob': log_prob,
                'done': done
            })
            
            state = next_state
        
        self.current_episode += 1
        return episode, episode_reward


    def compute_returns(self, episode):
        """
        Compute discounted returns (cumulative rewards) for an episode.
        Returns are computed backwards: G_t = r_t + gamma * G_{t+1}
        """
        returns = []
        G = 0.0
        
        # Compute returns backwards
        for transition in reversed(episode):
            reward = transition['reward']
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        return returns


    def play(self):
        """
        Collect one episode and add it to the dataset.
        When enough episodes are collected, train the model.
        """
        episode, episode_reward = self.collect_episode()
        self.episodes.append(episode)
        
        # Train when we have enough episodes
        if len(self.episodes) >= self.num_episodes:
            self.train()
            self.episodes = []  # Clear episodes after training
            self.current_episode = 0


    def train(self):
        """
        Train the policy network using REINFORCE algorithm.
        Uses collected episodes to compute policy gradient updates.
        """
        if len(self.episodes) == 0:
            print("Warning: No episodes collected, skipping training")
            return
        
        # Flatten all episodes into transitions
        all_states = []
        all_actions = []
        all_returns = []
        all_log_probs = []
        
        total_episode_length = 0
        for episode in self.episodes:
            returns = self.compute_returns(episode)
            total_episode_length += len(episode)
            
            for i, transition in enumerate(episode):
                all_states.append(transition['state'])
                all_actions.append(transition['action'])
                all_returns.append(returns[i])
                all_log_probs.append(transition['log_prob'])
        
        if len(all_states) == 0:
            print("Warning: No transitions collected, skipping training")
            return
        
        # Convert to tensors
        states = tf.convert_to_tensor(np.array(all_states), dtype=tf.float32)
        actions = tf.convert_to_tensor(np.array(all_actions), dtype=tf.int32)
        returns = tf.convert_to_tensor(np.array(all_returns), dtype=tf.float32)
        
        # Normalize returns (reduce variance) - handle case where all returns are similar
        returns_mean = tf.reduce_mean(returns)
        returns_std = tf.math.reduce_std(returns)
        
        # Debug: print raw returns info
        print(f"\n=== TRAINING DEBUG ===")
        print(f"Episodes: {len(self.episodes)}, Total transitions: {len(all_states)}, Avg episode length: {total_episode_length/len(self.episodes):.1f}")
        print(f"Raw returns: mean={float(returns_mean):.2f}, std={float(returns_std):.2f}, min={float(tf.reduce_min(returns)):.2f}, max={float(tf.reduce_max(returns)):.2f}")
        
        if returns_std > 1e-6:  # Only normalize if there's meaningful variance
            returns = (returns - returns_mean) / returns_std
            print(f"Normalized returns: mean={float(tf.reduce_mean(returns)):.4f}, std={float(tf.math.reduce_std(returns)):.4f}")
        else:
            # If all returns are similar, use raw returns (don't normalize to zero)
            # Just scale them down a bit to prevent huge gradients
            returns = returns / 10.0  # Scale down but don't zero them out
            print(f"Scaled returns (std too small): mean={float(tf.reduce_mean(returns)):.4f}, std={float(tf.math.reduce_std(returns)):.4f}")
        print(f"=====================\n")
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices({
            'states': states,
            'actions': actions,
            'returns': returns
        })
        dataset = dataset.shuffle(buffer_size=len(all_states)).batch(self.batch_size)
        
        # Training loop
        for epoch in range(self.epochs):
            epoch_loss = tf.keras.metrics.Mean()
            epoch_entropy = tf.keras.metrics.Mean()
            first_batch = True
            
            for batch in dataset:
                batch_states = batch['states']
                batch_actions = batch['actions']
                batch_returns = batch['returns']
                
                with tf.GradientTape() as tape:
                    # Get policy logits
                    logits = self.model(batch_states, isTesting=False)
                    logits = tf.reshape(logits, (tf.shape(logits)[0], -1))
                    
                    # Get action masks for each state in batch
                    batch_masked_logits = []
                    for i in range(tf.shape(batch_states)[0]):
                        # Reconstruct revealed mask from state encoding
                        # Channel 9 is the unknown channel: 1 = unknown (clickable), 0 = revealed
                        unknown_channel = batch_states[i, :, :, 9]
                        # unknown_channel > 0.5 means unknown (clickable)
                        revealed_mask = (unknown_channel < 0.5).numpy()  # True where revealed
                        action_mask = make_action_mask(revealed_mask, flatten=True)
                        mask_tf = tf.convert_to_tensor(action_mask, dtype=tf.bool)
                        
                        # Mask logits (large positive for invalid actions so they get low prob after negation)
                        masked_logits = tf.where(
                            mask_tf,
                            logits[i],
                            tf.fill(tf.shape(logits[i]), 1e9)
                        )
                        batch_masked_logits.append(masked_logits)
                    
                    batch_masked_logits = tf.stack(batch_masked_logits)
                    
                    # Negate logits: lower (safer) -> higher probability
                    negated_logits = -batch_masked_logits
                    
                    # Get log probabilities
                    log_probs = tf.nn.log_softmax(negated_logits)
                    
                    # Get log probability of selected actions
                    batch_size = tf.shape(batch_actions)[0]
                    indices = tf.stack([tf.range(batch_size), batch_actions], axis=1)
                    selected_log_probs = tf.gather_nd(log_probs, indices)
                    
                    # Policy gradient loss: -log_prob * return
                    # We want to maximize expected return, so minimize negative log_prob * return
                    policy_loss = -tf.reduce_mean(selected_log_probs * batch_returns)
                    
                    # Entropy bonus for exploration (positive entropy is good, so we subtract it from loss)
                    probs = tf.nn.softmax(negated_logits)
                    entropy = -tf.reduce_sum(probs * log_probs, axis=1)
                    entropy_bonus = -self.entropy_coef * tf.reduce_mean(entropy)
                    
                    # Total loss (we minimize this, so negative of what we want to maximize)
                    loss = policy_loss + entropy_bonus
                    
                    # Debug first batch of first epoch
                    if epoch == 0 and first_batch:
                        print(f"First batch: policy_loss={float(policy_loss):.6f}, entropy_bonus={float(entropy_bonus):.6f}, loss={float(loss):.6f}")
                        print(f"  selected_log_probs range: [{float(tf.reduce_min(selected_log_probs)):.4f}, {float(tf.reduce_max(selected_log_probs)):.4f}]")
                        print(f"  batch_returns range: [{float(tf.reduce_min(batch_returns)):.4f}, {float(tf.reduce_max(batch_returns)):.4f}]")
                        print(f"  entropy range: [{float(tf.reduce_min(entropy)):.4f}, {float(tf.reduce_max(entropy)):.4f}]")
                
                # Compute gradients and update
                gradients = tape.gradient(loss, self.model.trainable_variables)
                
                # Check for None gradients
                if any(g is None for g in gradients):
                    print("WARNING: Some gradients are None!")
                    continue
                
                # Clip gradients to prevent exploding gradients
                gradients = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in gradients]
                
                # Check gradient norms for first batch
                if epoch == 0 and first_batch:
                    grad_norms = [tf.norm(g).numpy() if g is not None else 0.0 for g in gradients]
                    print(f"  Gradient norms: {[f'{n:.6f}' for n in grad_norms[:3]]}...")
                    first_batch = False
                
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                
                epoch_loss.update_state(loss)
                epoch_entropy.update_state(tf.reduce_mean(entropy))
            
            if epoch % 2 == 0 or epoch == self.epochs - 1:
                loss_val = float(epoch_loss.result().numpy())
                entropy_val = float(epoch_entropy.result().numpy())
                # Print more info for debugging
                avg_return = float(tf.reduce_mean(returns).numpy())
                print(f"Epoch {epoch + 1}, Loss: {loss_val:.4f}, Entropy: {entropy_val:.4f}, Avg Return: {avg_return:.4f}")
                if WAND_AVAILABLE:
                    try:
                        _wandb.log({
                            "train/epoch_loss": loss_val,
                            "train/entropy": entropy_val,
                            "epoch": epoch + 1
                        })
                    except Exception:
                        pass
        
        # Save checkpoint
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
        
        Args:
            board: Current player board state
            deterministic: If True, use greedy action. If False, sample from policy.
        
        Returns:
            action: Selected action index, or None if no valid actions
        """
        # Get revealed mask for encoding
        revealed_mask = (board != BrickType.UNKNOWN)
        
        # Encode the current state
        board_encoded = encode_full(
            board,
            revealed_mask,
            channels_first=False
        )
        
        # Get policy logits
        input_tensor = tf.convert_to_tensor(board_encoded, dtype=tf.float32)
        input_tensor = tf.expand_dims(input_tensor, axis=0)  # (1, H, W, C)
        
        logits = self.model(input_tensor, isTesting=True)
        logits = tf.reshape(logits, (1, -1))
        
        # Create action mask
        action_mask = make_action_mask(revealed_mask, flatten=True)
        mask_tf = tf.convert_to_tensor(action_mask, dtype=tf.bool)
        
        # Mask out invalid actions
        masked_logits = tf.where(
            mask_tf[None, :],
            logits,
            tf.fill(tf.shape(logits), 1e9)  # Large positive (low prob after negation)
        )
        
        # Check if there are any valid actions
        if not tf.reduce_any(mask_tf):
            return None
        
        # Negate logits: lower (safer) -> higher probability
        negated_logits = -masked_logits[0]
        
        if deterministic:
            # Greedy action: choose action with lowest logit (safest)
            action = int(tf.argmin(masked_logits[0]).numpy())
        else:
            # Sample from policy
            probs = tf.nn.softmax(negated_logits)
            action = tf.random.categorical(tf.expand_dims(tf.math.log(probs + 1e-8), 0), 1)[0, 0]
            action = int(action.numpy())
        
        return action


    def validate(self, num_games=100):
        """
        Plays `num_games` complete Minesweeper games using the current policy.
        Returns win rate and average steps survived.
        """
        wins = 0
        total_steps = 0
        total_reward = 0.0

        for _ in range(num_games):
            # Reset environment
            state, info = self.env.reset()
            done = False
            steps = 0
            episode_reward = 0.0

            while not done:
                action = self.choose_action(state, deterministic=True)  # Use deterministic for evaluation
                
                if action is None:
                    break

                self.env.button = 'left'
                state, reward, done, info = self.env.step(action)
                steps += 1
                episode_reward += reward

            if info.get('status') == 'win':
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
                _wandb.log({
                    "eval/win_rate": win_rate,
                    "eval/avg_steps": avg_steps,
                    "eval/avg_reward": avg_reward
                })
            except Exception:
                pass

        return win_rate, avg_steps
