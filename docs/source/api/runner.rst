Runner
======

The runner is a standalone Python process (entry point ``sharpie-runner``)
that connects to the web server over WebSockets and drives the RL
experiment loop.  It has **no direct database access**; all persistence
goes through the web server.

Entry point: :mod:`sharpie.runner.manage`.

Startup sequence
----------------

.. code-block:: text

   main()
   │
   ├── connect to Queue WS  (/runner/connection)
   │   └── poll: send {"status": "idle"}, wait for {"experiment": …, "room": …}
   │
   └── spawn Process → start_experiment()
       │
       ├── connect to Experiment WS  (/experiment/<link>/run/<room>)
       ├── load_episode()          ← receive env / agent / experiment settings
       └── run_episode()
           ├── load_environment()  ← importlib load of environment file
           ├── load_policies()     ← importlib load of each policy file
           └── step loop
               ├── send_message()         → observation frame to web server
               ├── receive_message()      ← batched participant actions
               ├── get_policy_actions()   ← policy.predict()
               ├── override_actions()     ← apply human overrides
               ├── override_rewards()     ← apply human reward feedback
               ├── env.step()
               └── train_policies()       ← policy.update() (if interval hit)


.. autosummary::
    :toctree: _autosummary
    :recursive:
    :signatures: short

    sharpie.runner
