from django.shortcuts import render
from amaze import (
    Maze,
    Robot,
    MazeWidget,
    Simulation,
    qt_application,
    load,
    amaze_main,
)
from . import q_learning_env

def index(request):
    if request.method =='POST':
        if(not request.session.has_key('iteration')):
            request.session['iteration'] = 1
        elif(request.POST.get("reset", False)):
            request.session['iteration'] = 1
        else:
            request.session['iteration'] += 1

        FOLDER = "./static/AMaze/visualization"
        WIDTH = 512  # Of generated images

        # Define a seed-less maze and get the resolved name
        maze_str = "10x10"
        maze = Maze.from_string(maze_str)
        maze_str = maze.to_string()

        # Draw the maze
        app = qt_application()
        maze_img = f"{FOLDER}/{maze_str}.png"
        MazeWidget.static_render_to_file(maze=Maze.from_string(maze_str), path=maze_img, size=WIDTH, robot=False, solution=True, dark=True)
        
        # Have an agent move around in the maze ...
        agent_path = "./static/AMaze/unicursive_tabular.zip"
        controller = load(agent_path)
        simulation = Simulation(maze, Robot.BuildData.from_string("DD"), save_trajectory=True)
        simulation.run(controller)

        # ... and print its trajectory
        agent_name = agent_path.split("/")[-1].split(".")[0]
        trajectory_img = f"{FOLDER}/{agent_name}_{maze_str}.png"
        MazeWidget.plot_trajectory(
            simulation=simulation,
            size=WIDTH,
            path=trajectory_img,
        )

        return render(request, "AMaze/index.html", {'iteration': request.session['iteration'], "success": simulation.success(), "cumulative_reward": simulation.cumulative_reward(), "normalized_reward": simulation.normalized_reward(), "agent_name": agent_name, "maze_str": maze_str})
    else:
        return render(request, "AMaze/index.html")

def q_learning(request):
    if request.method =='POST':
        if(not request.session.has_key('stepsize') or request.session['stepsize']==''):
            request.session['stepsize'] = 1

        if(not request.session.has_key('iteration') or request.POST.get("start", False)):
            request.session['iteration'] = int(request.POST.get("stepsize", 1))
            request.session['policy_file'] = None
        elif(request.POST.get("reset", False)):
            request.session['iteration'] = int(request.POST.get("stepsize", 1))
            request.session['policy_file'] = None
        elif(request.POST.get("continue", False)):
            request.session['iteration'] += int(request.POST.get("stepsize", 1))
        
        if(not request.POST.get("evaluate", False)):
            request.session['policy_file'] = q_learning_env.train(int(request.POST.get("stepsize", 1)), request.session['policy_file'])
        (simulation, maze_str) = q_learning_env.eval(request.session['policy_file'])


        return render(request, "AMaze/q_learning.html", {'iteration': request.session['iteration'], "success": simulation.success(), "cumulative_reward": simulation.cumulative_reward(), "normalized_reward": simulation.normalized_reward(), "agent_name": 'Q-learning', "maze_str": maze_str, "stepsize": int(request.POST.get("stepsize", 1))})
    else:
        return render(request, "AMaze/q_learning.html")