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
import pickle



def index_(request):
    return render(request, "AMaze/index.html")



def stored_policy_(request):
    request.session['example_policy'] = True
    request.session['policy_file'] = "./static/AMaze/unicursive_tabular.zip"
    request.session['reward_file'] = None
    request.session['iteration'] = 0
    if not request.session.has_key('stepsize'):
        request.session['stepsize'] = 1

    return evaluate_(request)



def restart_(request):
    request.session['example_policy'] = False
    request.session['policy_file'] = None
    request.session['reward_file'] = None
    request.session['iteration'] = 0
    if not request.session.has_key('stepsize'):
        request.session['stepsize'] = 1

    return continue_(request)



def continue_(request):
    request.session['example_policy'] = False
    if request.method =='POST':
        request.session['stepsize'] = request.POST.get("stepsize", request.session['stepsize'])

    request.session['iteration'] += int(request.session['stepsize'])

    if request.session['reward_file']:
        with open(request.session['reward_file'], 'rb') as f:
            ras = pickle.load(f)
        for i, (state, action, reward, state_, action_) in enumerate(ras):
            ras[i] = [state, action, float(request.POST.get(str(i), reward)), state_, action_]
        q_learning_env.learn(request.session['policy_file'], ras)

    for i in range(int(request.session['stepsize'])-1):
        (_, _, ras, _, request.session['policy_file']) = q_learning_env.train(False, request.session['policy_file'])
        q_learning_env.learn(request.session['policy_file'], ras)
    (simulation, maze_str, ras, request.session['reward_file'], request.session['policy_file']) = q_learning_env.train(True, request.session['policy_file'])

    return render(request, "AMaze/q_learning.html", {'iteration': request.session['iteration'],
                                                     "success": simulation.success(),
                                                     "cumulative_reward": simulation.cumulative_reward(),
                                                     "normalized_reward": simulation.normalized_reward(),
                                                     "agent_name": 'Q-learning', "maze_str": maze_str,
                                                     "stepsize": request.session['stepsize'],
                                                     'ras': ras, 'ras_max':len(ras)})




def evaluate_(request):
    (simulation, maze_str, ras) = q_learning_env.eval(request.session['policy_file'])

    return render(request, "AMaze/q_learning.html", {'evaluate': request.session['example_policy'],
                                                     'iteration': request.session['iteration'],
                                                     "success": simulation.success(),
                                                     "cumulative_reward": simulation.cumulative_reward(),
                                                     "normalized_reward": simulation.normalized_reward(),
                                                     "agent_name": 'Q-learning', "maze_str": maze_str,
                                                     "stepsize": request.session['stepsize'],
                                                     'ras': ras, 'ras_max': len(ras)})