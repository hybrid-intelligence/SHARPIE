Web Server
==========

The web server is a `Django / Daphne`_ ASGI application.
It serves the participant-facing UI over HTTP(S) and brokers real-time
communication between participants and the runner over two WebSocket
channels:

* **Queue WebSocket** (``/runner/connection``) — polled by the runner to
  receive pending session assignments.
* **Experiment WebSocket** (``/experiment/<link>/run/<room>``) — carries
  observations, actions and rewards for the duration of a session.

The application is composed of five Django apps:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - App
     - Purpose
   * - ``home``
     - Landing page; no models.
   * - ``accounts``
     - User authentication, consent forms and participant records.
   * - ``experiment``
     - Core configuration models: environments, policies, agents and
       experiments.
   * - ``data``
     - Per-session logging: sessions, episodes and step-level records.
   * - ``runner``
     - Runner registration and status tracking.

.. _Django / Daphne: https://docs.djangoproject.com/en/stable/

.. autosummary::
    :toctree: _autosummary
    :recursive:

    sharpie.webserver
