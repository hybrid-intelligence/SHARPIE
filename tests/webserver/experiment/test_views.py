"""Tests for the experiment pages."""
from django.urls import reverse

from sharpie.webserver.data.models import Session


def test_configuration_page_puts_the_participant_in_the_chosen_room(
        client, experiment, participant):
    """Submitting the configuration page creates the room's session and stores the chosen role."""
    client.force_login(participant)

    response = client.post(reverse('config', args=[experiment.link]),
                           {'room': 'room1', 'role': 'agent_0'})

    assert response.status_code == 302
    room_session = Session.objects.get(experiment=experiment, room='room1')
    assert room_session.participants.filter(user=participant).exists()
    assert client.session['role'] == 'agent_0'
