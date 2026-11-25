"""Browser tests for the game page against SHARPIE's real game room, served by Daphne."""
import json

import pytest
from channels.testing import ChannelsLiveServerTestCase
from django.test import override_settings
from playwright.async_api import async_playwright, expect
from websockets.asyncio.client import connect

IN_MEMORY_CHANNEL_LAYER = {'default': {'BACKEND': 'channels.layers.InMemoryChannelLayer'}}


@pytest.mark.daphne
@override_settings(CHANNEL_LAYERS=IN_MEMORY_CHANNEL_LAYER)
class GamePageWithGameRoomTests(ChannelsLiveServerTestCase):
    """The game page talking to SHARPIE's real game room, with a script playing the runner."""

    @pytest.fixture(autouse=True)
    def _prepare(self, participant_session_id, runner, room_path, jpeg_frame):
        """Give each test a joined participant, a registered runner, the room's path and a frame."""
        self.session_id = participant_session_id
        self.runner = runner
        self.room_path = room_path
        self.frame = jpeg_frame

    async def test_frame_and_action_travel_between_page_and_runner(self):
        """The page shows a frame from the runner, and the runner receives the page's action."""
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            page = await browser.new_page()
            await page.context.add_cookies([{'name': 'sessionid', 'value': self.session_id,
                                             'url': self.live_server_url}])
            async with page.expect_websocket() as page_websocket:
                await page.goto(self.live_server_url + self.room_path)
            await page_websocket.value

            async with connect(
                    self.live_server_ws_url + self.room_path,
                    additional_headers={'authorization': self.runner.connection_key}) as runner:
                await runner.send(json.dumps({'type': 'broadcast', 'image': self.frame}))
                await expect(page.locator('#image')).to_have_attribute(
                    'src', f'data:image/jpeg;base64,{self.frame}')
                message = json.loads(await runner.recv())
            await browser.close()

        self.assertEqual(message['batch_actions'], {'agent_0': 0})
