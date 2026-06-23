Engineering Principles
======================

| The purpose of SHARPIE is to be a solution serving a wide variety of research questions across different interaction paradigms and learning modalities.
| We therefore aim to follow the following principles in the design and implementation of SHARPIE

+++++++
Modular
+++++++
| Human-AI interaction research is rapidly evolving, requiring a framework that can adapt without needing a complete overhaul.
| SHARPIE aims for a strict separation of concerns by decoupling the core RL environments, the web server logic, the participant-facing web interface, and backend logging utilities.
| It thereby allows researchers to swap out algorithm libraries, modify user interfaces, or plug in new interaction paradigms independently, ensuring the codebase remains clean and maintainable.

+++++++++++++++++++++++++++
Customizable & Configurable
+++++++++++++++++++++++++++
| Every researcher, and indeed, every individual study has slightly different requirements on the interaction paradigm, the logging, and the deployment of their software.
| However, there are still many across different researchers looking into the interaction of humans with (adaptive) sequential decision-making agents.
| We therefore aim to build SHARPIE in a flexible way, with hooks for extending and customizing SHARPIE.

New functionalities can be adopted by SHARPIE once a reasonable mass of support for that functionality is attained. To request/suggest a new feature, please open an issue in the issue tracker to motivate the feature, where we will follow an (informal) approval process.

++++++++++++++++++++++++
Multi-agent, Multi-human
++++++++++++++++++++++++
| Real-world human-AI interaction is rarely limited to a single agent and a single human user.
| SHARPIE is built from the ground up to support multi-agent multi-human experiments, including multi-agent systems, multi-user studies, and collaborative human-AI teams composed of multiple and heterogenous agents.
| The underlying communication protocols and state representations are inherently designed to scale across multiple participants and autonomous entities simultaneously.

+++++++++++++++++
Easily deployable
+++++++++++++++++
| Transitioning an experiment from a local development environment to a live participant study should be seamless.
| We prioritize out-of-the-box compatibility with popular cloud providers and participant recruitment platforms.

++++++++++++++++
Production ready
++++++++++++++++
| We want the excitement to come from your experimental result, not from whether your experiment runs.
| Human-subject experimentation requires software that is stable, reliable, and tested.
| SHARPIE relies on established frameworks and implements industry-level quality control and testing.

+++++++++++++++++++
Simple over complex
+++++++++++++++++++
| Given the interdisciplinary nature of human-AI research, SHARPIE must remain accessible to researchers from diverse technical backgrounds.
| We favor explicit code, straightforward architectures, and minimal abstraction layers over clever or overly intricate engineering designs.
| We aim to choose the path that yields the most legible code and the lowest cognitive load for the community in usage, maintenance and development.
