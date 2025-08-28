import numpy as np
import random
import matplotlib.pyplot as plt
import pygame

# Defining agent class
class AffSimAgent:
    # Initialize the Agent
    def __init__(self, num_states, num_actions, alpha, gamma, epsilon, epsilon_decay):
        self.q_table = np.zeros((num_states, num_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

    # Define action selection (and call it Decision Making)
    def decision_making(self, state):
        random_number = random.random()
        exploratory = random_number < self.epsilon
        if exploratory:
            action = random.randint(0, 3)
        else:
            action = int(np.argmax(self.q_table[state]))
        return action, exploratory
    
    # Define Update Method (and call it Learning Mechanism)
    def learning_mechanism(self, state, action, reward, next_state):
        td = reward + self.gamma * np.max(self.q_table[next_state, ])
        td_error = td - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error




#environment
class Room1:
    def __init__(self):
        # Define the maze as a list of strings, F = floor, O=Obstacle, D = Dirt, E=Exit
        maze_layout = [
            "SFFFFFFF",
            "FFFFFFFF",
            "FDFOFFFF",
            "FFFFFOFF",
            "FFFOFFDF",
            "FOOFFFOF",
            "FOFFOFOF",
            "FFFOFFFE",
        ]

        self.grid_size = len(maze_layout)
        self.obstacles = set()
        self.dirt_patches = set()
        self.cleaned_dirt = set()
        self.start_pos = None
        self.goal_pos = None
        self.broken_obstacles = set()
        self.agent_orientation = 1  # Start facing down (0=up, 1=down, 2=left, 3=right)

        # Parse the maze
        for row_idx, row in enumerate(maze_layout):
            for col_idx, cell in enumerate(row.strip().upper()):
                pos = (row_idx, col_idx)
                if cell == "S":
                    self.start_pos = pos
                elif cell == "E":
                    self.goal_pos = pos
                elif cell == "O":
                    self.obstacles.add(pos)
                elif cell == "D":
                    self.dirt_patches.add(pos)

        self.current_pos = self.start_pos

    def load_images(self, tile_size):
        self.tile_size = tile_size
        self.image_assets = {
            "F": pygame.image.load("floor.png"),
            "D": pygame.image.load("dirt.png"),
            "O": pygame.image.load("obstacle.png"),
            "B": pygame.image.load("broken_obstacle.png"),
            "E": pygame.image.load("exit.png"),
            "S": pygame.image.load("floor.png"),
            "agent_0": pygame.image.load("agent_up.png"),
            "agent_1": pygame.image.load("agent_down.png"),
            "agent_2": pygame.image.load("agent_left.png"),
            "agent_3": pygame.image.load("agent_right.png")
        }

        for key in self.image_assets:
            self.image_assets[key] = pygame.transform.scale(
                self.image_assets[key], (tile_size, tile_size)
            )

    def render(self, screen):
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                pos = (row, col)

                if pos == self.goal_pos:
                    img = self.image_assets["E"]
                elif pos in self.broken_obstacles:
                    img = self.image_assets["B"]
                elif pos in self.obstacles:
                    img = self.image_assets["O"]
                elif pos in self.dirt_patches and pos not in self.cleaned_dirt:
                    img = self.image_assets["D"]
                else:
                    img = self.image_assets["F"]

                screen.blit(img, (col * self.tile_size, row * self.tile_size))

        # Draw agent with orientation
        row, col = self.current_pos
        agent_img = self.image_assets[f"agent_{self.agent_orientation}"]
        screen.blit(agent_img, (col * self.tile_size, row * self.tile_size))


    def step(self, action):
        self.agent_orientation = action  # ← TRACK ORIENTATION
        row, col = self.current_pos

        # Determine the intended new position
        if action == 0:    # Up
            new_pos = (row - 1, col)
        elif action == 1:  # Down
            new_pos = (row + 1, col)
        elif action == 2:  # Left
            new_pos = (row, col - 1)
        else:              # Right
            new_pos = (row, col + 1)

        # Check bounds
        if 0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size:
            self.current_pos = new_pos  # Move
        #else:
            #new_pos = self.current_pos  # Out of bounds — stay put

        #initialize these two
        reward = 0.0
        done = False

        # Determine reward
        if self.current_pos == self.goal_pos:
            reward = 1.0
            done = True
        elif self.current_pos in self.obstacles:
            reward = -1.0  # Penalty for stepping on obstacle
            self.broken_obstacles.add(self.current_pos)
            done = False
        elif self.current_pos in self.dirt_patches and self.current_pos not in self.cleaned_dirt:
            reward = 0.5
            self.cleaned_dirt.add(self.current_pos)
        else:
            reward = 0.0  # neutral reward
            done = False

        return self.get_state(), reward, done
    

    def get_state(self):
        row, col = self.current_pos
        return row * self.grid_size + col

    def reset(self):
        self.current_pos = self.start_pos
        self.cleaned_dirt = set()  # Reset dirt cleaned state
        return self.get_state()




# # Instantiate the environment
# env = Room1()
# state = env.reset()

# # Define a test sequence of actions
# # Actions: 0 = Up, 1 = Down, 2 = Left, 3 = Right
# # This sequence is arbitrary — you can modify to walk over dirt patches
# actions = [1, 1, 3, 1, 0, 3]  # Try walking into a dirt patch

# print("Testing agent-environment interaction:\n")
# for i, action in enumerate(actions):
#     state, reward, done = env.step(action)
#     print(f"Step {i+1}: Action={action}, State={state}, Reward={reward}, Done={done}")
#     if done:
#         print("Reached goal. Episode finished.")
#         break

# # Step onto same dirt again to test one-time reward
# print("\nRetesting dirt patch reward:")
# state = env.reset()
# # Manually step into same dirt twice
# env.step(1)  # Down
# state, reward, done = env.step(1)  # Down to dirt
# print(f"First visit to dirt: Reward={reward}")
# state, reward, done = env.step(0)  # Up
# state, reward, done = env.step(1)  # Down again
# print(f"Second visit to same dirt: Reward={reward}")


# Create environment and agent
env = Room1()
num_states = env.grid_size * env.grid_size
num_actions = 4

agent = AffSimAgent(
    num_states=num_states,
    num_actions=num_actions,
    alpha=0.1,           # learning rate
    gamma=0.9,           # discount factor
    epsilon=1.0,         # exploration rate
    epsilon_decay=0.995  # decay per episode
)

num_episodes = 500
max_steps = 100  # limit per episode

rewards_per_episode = []
saved_trajectories = {}  # episode_num: list of (state, orientation)

#Emotion simulation
class EmotionEngine:
    def __init__(self, relevance_threshold=0.5):
        self.relevance_threshold = relevance_threshold
        self.emotion_log = {"resilience": 0}

    def check_resilience_trigger(self, exploration_flag, td_error):
        frustration_trigger = abs(td_error) > self.relevance_threshold
        if exploration_flag and frustration_trigger:
            self.emotion_log["resilience"] += 1
            return True
        return False

    def get_emotion_counts(self):
        return self.emotion_log.copy()

emotion_engine = EmotionEngine(relevance_threshold=0.5)

first_resilience_episode = None  # to capture first resilience trigger
resilience_trajectory = []       # store its trajectory if triggered
resilience_episodes = []
resilience_trajectories = {}  # episode_num: trajectory





for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    trajectory = []

    for step in range(max_steps):
        action, exploration_flag = agent.decision_making(state)
        next_state, reward, done = env.step(action)

        #trajectory.append((env.current_pos, env.agent_orientation))


        # TD Error calculation (mimic learning_mechanism)
        td_target = reward + agent.gamma * np.max(agent.q_table[next_state])
        td_error = td_target - agent.q_table[state, action]
        agent.q_table[state, action] += agent.alpha * td_error

        trajectory.append({
            "position": env.current_pos,
            "orientation": env.agent_orientation,
            "td_error": td_error,
            "exploration": exploration_flag
        })


        # Emotion check (call only once!)
        resilience_triggered = emotion_engine.check_resilience_trigger(exploration_flag, td_error)
        if resilience_triggered:
            print(f"[Episode {episode+1}] Resilience expressed at step {step+1}")
            trajectory[-1]["resilience_triggered"] = True  # mark that this specific step triggered it

            if episode + 1 not in resilience_episodes:  # Prevent duplicates if triggered more than once
                resilience_episodes.append(episode + 1)
                resilience_trajectories[episode + 1] = trajectory.copy()



        state = next_state  
        total_reward += reward

        if done:
            break


    # Save trajectory if it's a 100th episode ✅ (INSIDE the loop)
    if (episode + 1) % 100 == 0:
        saved_trajectories[episode + 1] = trajectory.copy()

    agent.epsilon *= agent.epsilon_decay
    agent.epsilon = max(agent.epsilon, 0.05)
    rewards_per_episode.append(total_reward)

    if (episode + 1) % 50 == 0:
        print(f"Episode {episode+1}: Total Reward = {total_reward:.2f}, Epsilon = {agent.epsilon:.3f}")


def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


# Calculate smoothed rewards
smoothed_rewards = moving_average(rewards_per_episode, window_size=10)

# Plot both
plt.figure(figsize=(10, 6))
plt.plot(rewards_per_episode, label="Raw Total Reward", alpha=0.4)
plt.plot(range(9, len(smoothed_rewards)+9), smoothed_rewards, label="Smoothed (10-ep MA)", linewidth=2)

# Plot resilience points
if resilience_episodes:
    resilience_rewards = [rewards_per_episode[i - 1] for i in resilience_episodes]
    plt.scatter(resilience_episodes, resilience_rewards, color='red', label="Resilience  Triggered", zorder=5)
                
plt.title("Total Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.grid(True)
plt.show()






# Merge resilience and milestone trajectories
combined_trajectories = dict(saved_trajectories)  # start from 100th-episode milestones
combined_trajectories.update(resilience_trajectories)  # add resilience episodes

# Sort by episode number
sorted_combined = dict(sorted(combined_trajectories.items()))


def render_multiple_trajectories(trajectory_dict, emotion_log_dict, resilience_episode_set=None):
    if resilience_episode_set is None:
        resilience_episode_set = set()
    pygame.init()
    tile_size = 64
    env = Room1()
    env.load_images(tile_size)

    grid_size_px = env.grid_size * tile_size
    screen = pygame.display.set_mode((grid_size_px + 200, grid_size_px))  # Extra space on right

    pygame.display.set_caption("AffSim Agent")
    font = pygame.font.SysFont(None, 32)  # Use default font

    clock = pygame.time.Clock()

    for episode, trajectory in sorted(trajectory_dict.items()):
        resilience_flag = episode in resilience_episode_set
        caption = f"AffSim Agent - Episode {episode}"
        if resilience_flag:
            caption += " [Resilience Triggered]"
        pygame.display.set_caption(caption)
        step_index = 0
        env.reset()
        total_reward = 0

        while step_index < len(trajectory):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            step_data = trajectory[step_index]
            pos = step_data["position"]
            orientation = step_data["orientation"]
            td_error = step_data.get("td_error", 0.0)
            exploration = step_data.get("exploration", False)
            resilience_step_triggered = step_data.get("resilience_triggered", False)


            env.current_pos = pos
            env.agent_orientation = orientation


            # Calculate reward based on current tile (same as in step logic)
            if pos == env.goal_pos:
                step_reward = 1.0
            elif pos in env.obstacles and pos not in env.broken_obstacles:
                step_reward = -1.0
                env.broken_obstacles.add(pos)
            elif pos in env.dirt_patches and pos not in env.cleaned_dirt:
                step_reward = 0.5
                env.cleaned_dirt.add(pos)
            else:
                step_reward = 0.0

            total_reward += step_reward

            # Render environment
            if resilience_flag:
                screen.fill((30, 0, 30))  # Dark purple for resilience episodes
            else:
                screen.fill((0, 0, 0))    # Normal black

            env.render(screen)
            # Overlay red flash on agent tile if resilience triggered
            if resilience_step_triggered:
                row, col = env.current_pos
                flash_surface = pygame.Surface((tile_size, tile_size), pygame.SRCALPHA)
                flash_surface.fill((255, 0, 0, 120))  # Semi-transparent red
                screen.blit(flash_surface, (col * tile_size, row * tile_size))

            # Render reward counter on the right
            reward_text = font.render(f"Reward: {total_reward:.2f}", True, (255, 255, 255))
            screen.blit(reward_text, (grid_size_px + 20, 20))

            # Optional: show current step
            step_text = font.render(f"Step: {step_index+1}", True, (180, 180, 180))
            screen.blit(step_text, (grid_size_px + 20, 60))

            # TD Error
            td_text = font.render(f"TD Error: {td_error:.3f}", True, (255, 200, 200))
            screen.blit(td_text, (grid_size_px + 20, 100))

            # Exploration flag
            explore_text = font.render(f"Exploring: {'Yes' if exploration else 'No'}", True, (200, 255, 200))
            screen.blit(explore_text, (grid_size_px + 20, 140))

            # Display emotion counter(s)
            y_offset = 120
            for emotion, count in emotion_log_dict.items():
                emo_text = font.render(f"{emotion.title()}: {count}", True, (200, 200, 255))
                screen.blit(emo_text, (grid_size_px + 20, y_offset))
                y_offset += 30

            if resilience_step_triggered:
                resilience_msg = font.render("Resilience Triggered!", True, (255, 100, 100))
                screen.blit(resilience_msg, (grid_size_px + 20, y_offset + 10))  # below last stat

            pygame.display.flip()
            clock.tick(5)  # Adjust playback speed

            step_index += 1

        pygame.time.wait(1000)

    pygame.quit()

render_multiple_trajectories(
    sorted_combined,
    emotion_engine.get_emotion_counts(),
    resilience_episode_set=set(resilience_episodes)
)


# # Also render first resilience episode if it occurred
# if first_resilience_episode is not None:
#     print(f"\nRendering first resilience episode: {first_resilience_episode}")
#     render_multiple_trajectories(
#         {first_resilience_episode: resilience_trajectory},
#         emotion_engine.get_emotion_counts()
#     )
# else:
#     print("\nNo resilience episode occurred during training.")


