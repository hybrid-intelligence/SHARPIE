FAQ (Frequently Asked Questions)
=====

How long will SHARPIE be supported?
--------------------------------

SHARPIE is currently under active development we will continue do so until the end of the `HI center <https://www.hybrid-intelligence-centre.nl/>`_ in 2029.

Can SHARPIE integrate with environment X?
--------------------------------

SHARPIE integrates with any environment that implements the `Gymnasium API <https://gymnasium.farama.org/api/env/>`_: SHARPIE needs access to `step()`, `reset()` and `render()` functions. Since SHARPIE runs the environment on a back-end server rather than in the browser, it supports any environment with Python bindings to these functions.

What are the computational requirements for SHARPIE?
--------------------------------

SHARPIE itself does not come with strict computational requirements, we suggest that you follow the computational requirements of your environment and RL model of choice.

Can I use SHARPIE to run experiments involving participants from different continents?
--------------------------------

It depends. Having participants located in different continents in the same room is likely to cause latency issues that are fundamental to the nature of cross-continental networking. Separating participants per room may be a viable alternative if your experimental setup support this.

Why did you develop SHARPIE?
--------------------------------

We developed SHARPIE to facilitate experiments involving RL agents and humans. We believe that the study of this interaction is crucial to establishing artificial intelligence(s) that do not replace human intellect but instead expand it. We thereby want to put humans at the center, and change the course of the ongoing AI revolution.

Can SHARPIE be used for education and outreach?
--------------------------------

SHARPIE can be used for educational purposes and outreach. However, it is currently still under active development. We plan to develop educational materials on hybrid human-AI systems and experiments once the platform stabilizes.