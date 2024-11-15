import math
import pathlib
import random
import shutil
import time
import random

from amaze.simu.controllers.tabular import TabularController
from amaze import Maze, Robot, Simulation, InputType, OutputType, StartLocation, MazeWidget, qt_application

ALPHA, GAMMA = 0.1, 0.5
MAZE_STR = "20x20_C1"
FOLDER = "./static/AMaze/visualization"
ROBOT = Robot.BuildData(inputs=InputType.DISCRETE, outputs=OutputType.DISCRETE)


def train(n, policy_file=None):
    start_time = time.time()

    if(policy_file):
        policy = TabularController(robot_data=ROBOT, epsilon=0.1*random.random(), seed=0).load(policy_file)
    else:
        policy = TabularController(robot_data=ROBOT, epsilon=0.1*random.random(), seed=0)

    steps = [0, 0]

    for i in range(n):
        maze = Maze.from_string(MAZE_STR)
        simulation = Simulation(maze, ROBOT)
        
        q_train(simulation, policy)

        steps[0] += simulation.timestep
    
    policy.save("./static/AMaze/policy")

    print(
        f"Training took {time.time() - start_time:.2g} seconds for:\n"
        f" > {steps[0]} training steps\n"
        f" > {steps[1]} evaluating steps"
    )

    return "./static/AMaze/policy.zip"

def eval(policy_file):
    policy = TabularController(robot_data=ROBOT, epsilon=0.1*random.random(), seed=0).load(policy_file)

    maze = Maze.from_string(MAZE_STR)
    simulation = Simulation(maze, ROBOT, save_trajectory=True)

    q_eval(simulation, policy)

    app = qt_application()
    maze_img = f"{FOLDER}/{MAZE_STR}.png"
    MazeWidget.static_render_to_file(maze=maze, path=maze_img, size=500, robot=False, solution=True, dark=True)

    trajectory_img = f"{FOLDER}/Qlearning_{MAZE_STR}.png"
    MazeWidget.plot_trajectory(
        simulation=simulation,
        size=500,
        path=trajectory_img,
    )

    return (simulation, MAZE_STR)


def q_train(simulation, policy):
    state = simulation.generate_inputs().copy()
    action = policy(state)

    while not simulation.done():
        reward = simulation.step(action)
        state_ = simulation.observations.copy()
        action_ = policy(state)
        policy.q_learning(
            state, action, reward, state_, action_, alpha=ALPHA, gamma=GAMMA
        )
        state, action = state_, action_

    return simulation.robot.reward


def q_eval(simulation, policy):
    action = policy.greedy_action(simulation.observations)
    while not simulation.done():
        simulation.step(action)
        action = policy.greedy_action(simulation.observations)

    return simulation.robot.reward


def evaluate_generalization(policy):
    policy.epsilon = 0
    rng = random.Random(0)

    n = 1000
    rewards = []

    print()
    print("=" * 80)
    print("Testing for generalization")

    print("\n-- Navigation", "-" * 66)
    _log_format = "\r[{:6.2f}%] normalized reward: {:.1g} for {}"

    for i in range(n):
        maze_data = Maze.BuildData(
            width=rng.randint(10, 30),
            height=rng.randint(10, 20),
            seed=rng.randint(0, 10000),
            unicursive=True,
            start=rng.choice([sl for sl in StartLocation]),
            p_lure=0,
            p_trap=0,
        )
        maze = Maze.generate(maze_data)
        simulation = Simulation(maze, ROBOT)
        simulation.run(policy)
        reward = simulation.normalized_reward()
        rewards.append(reward)
        print(
            _log_format.format(100 * (i + 1) / n, reward, maze_data.to_string()),
            end="",
            flush=True,
        )
    print()

    avg_reward = sum(rewards) / n
    optimal = " (optimal)" if math.isclose(avg_reward, 1) else ""
    print(f"Average score of {avg_reward}{optimal} on {n} random mazes")

    print("\n-- Inputs", "-" * 70)
    print(Simulation.inputs_evaluation(FOLDER, policy, signs=dict()))

    print("=" * 80)


def main(is_test=False):
    if FOLDER.exists():
        shutil.rmtree(FOLDER)
    FOLDER.mkdir(parents=True, exist_ok=False)

    policy = train()

    policy_file = policy.save(
        FOLDER.joinpath("policy"), dict(comment="Can solve unicursive mazes")
    )
    print("Saved optimized policy to", policy_file)

    evaluate_generalization(policy)


if __name__ == "__main__":
    main()
