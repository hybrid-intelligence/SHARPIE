"""Helper classes and functions for WebSocket consumers."""

import os
import base64
import pickle
import lzma

from django.db import transaction
from django.db.models import F, Prefetch
from django.db.models.functions import Now

from data.models import Session


class RunConsumerHelpers:
    """Synchronous helper methods for RunConsumer.

    These methods perform database operations and are wrapped by async methods
    in RunConsumer using database_sync_to_async.
    """

    def _do_increment(self):
        """Atomically increment participant count and return True if all connected."""
        with transaction.atomic():
            # Lock the session row and update
            session = Session.objects.select_for_update().get(pk=self.session.pk)
            session.connected_participants = F('connected_participants') + 1
            session.save(update_fields=['connected_participants'])

            # Refresh to get the actual value
            session.refresh_from_db()
            participants_count = session.participants.count()

            return session.connected_participants == participants_count

    def _fetch_episode_info(self):
        """Fetch episode completion info from database."""
        session = Session.objects.prefetch_related('experiment').get(pk=self.session.pk)
        completed_count = session.episodes.filter(completed=True).count()
        return {
            'completed_count': completed_count,
            'number_of_episodes': session.experiment.number_of_episodes,
            'redirect_url': session.experiment.redirect_url
        }

    def _fetch_settings_data(self):
        """Fetch all settings data from database."""
        # Refresh session with prefetched relations
        session = Session.objects.prefetch_related(
            'experiment__environment',
            Prefetch('experiment__agents')
        ).get(pk=self.session.pk)

        experiment = session.experiment
        environment = experiment.environment

        # Build environment message
        env_message = {
            'files': {},
            'metadata': environment.metadata or {},
        }
        for name, filepath in environment.filepaths.items():
            if self.runner.ip_address != '127.0.0.1':
                full_path = os.path.join('..', 'runner', filepath)
                with open(full_path, 'r') as f:
                    code = f.read()
                env_message['files'][name] = {'path': filepath, 'content': code}
            else:
                env_message['files'][name] = {'path': filepath, 'content': None}

        # Build agents message
        agents_message = {}
        for agent in experiment.agents.all():
            agents_message[agent.role] = {
                'participant': agent.participant,
                'keyboard_inputs': agent.keyboard_inputs,
                'multiple_keyboard_inputs': agent.multiple_keyboard_inputs,
                'textual_inputs': agent.textual_inputs,
                'inputs_type': agent.inputs_type
            }
            if agent.policy:
                agents_message[agent.role]['policy'] = {'checkpoint_interval': agent.policy.checkpoint_interval}
                agents_message[agent.role]['policy']['files'] = {}
                for name, filepath in agent.policy.filepaths.items():
                    if self.runner.ip_address != '127.0.0.1':
                        full_path = os.path.join('..', 'runner', filepath)
                        with open(full_path, 'r') as f:
                            code = f.read()
                        agents_message[agent.role]['policy']['files'][name] = {'path': filepath, 'content': code}
                    else:
                        agents_message[agent.role]['policy']['files'][name] = {'path': filepath, 'content': None}

        # Build experiment message
        exp_message = {
            'conda_environment': experiment.conda_environment,
            'target_fps': experiment.target_fps,
            'wait_for_inputs': experiment.wait_for_inputs
        }

        return {
            'environment': env_message,
            'agents': agents_message,
            'experiment': exp_message
        }

    def _do_session_update(self):
        """Update session status on runner disconnect."""
        try:
            self.session.refresh_from_db()
        except Exception:
            return

        self.session.connected_participants = 0

        # Get episode counts with prefetched experiment
        session = Session.objects.prefetch_related('experiment').get(pk=self.session.pk)
        completed_count = session.episodes.filter(completed=True).count()
        total_episodes = session.episodes.count()
        number_of_episodes = session.experiment.number_of_episodes

        if completed_count >= number_of_episodes:
            self.session.status = 'completed'
            self.session.end_time = Now()
        elif total_episodes >= number_of_episodes:
            self.session.status = 'aborted'
            self.session.end_time = Now()
        else:
            self.session.status = 'ready'

        self.session.save()


def decode_data(data):
    """Decode data that may contain LZMA-compressed pickled arrays.

    Recursively processes dictionaries and lists, converting any
    compressed arrays back to numpy arrays.
    """
    if isinstance(data, dict):
        # Check if this is a LZMA-compressed array
        if data.get("__lzma__") is True:
            compressed = base64.b64decode(data["data"])
            return pickle.loads(lzma.decompress(compressed))
        # Otherwise, recursively process dictionary values
        return {k: decode_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [decode_data(v) for v in data]
    else:
        return data