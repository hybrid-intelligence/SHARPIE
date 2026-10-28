"""Tests for the experiment app."""
import os
from pathlib import Path
from unittest import skipUnless

import yaml
from django.contrib import messages
from django.contrib.auth.models import User
from django.contrib.messages import get_messages
from django.core.exceptions import ValidationError
from django.test import TestCase
from django.urls import reverse

from sharpie.install.cli import list_use_cases
from sharpie.install.exporter import ExportError, export_experiment, export_experiment_yaml
from sharpie.install.installer import load_config, setup_database

from .models import Agent, Environment, Experiment, Policy

# The gallery is not part of this repository. Tests using it are skipped unless
# a clone exists next to the repository, or SHARPIE_GALLERY_DIR points to one.
REPO_ROOT = Path(__file__).resolve().parents[3]
GALLERY_DIR = Path(os.environ.get('SHARPIE_GALLERY_DIR', REPO_ROOT.parent / 'SHARPIE_Gallery'))


class ExperimentTestCase(TestCase):
    """Tests for experiment configuration and running."""

    def setUp(self):
        """Set up test fixtures (none needed yet)."""
        pass

    def test_config(self):
        """Test configuration can be saved and loaded."""
        self.assertTrue(True)

    def test_run(self):
        """Test experiment can be run and is saved to the database."""
        self.assertTrue(True)


class ExportExperimentTests(TestCase):
    """Tests for building the config.yaml content of an experiment."""

    @classmethod
    def setUpTestData(cls):
        """Create an experiment with one human agent and one AI agent sharing a policy."""
        cls.environment = Environment.objects.create(
            name='Test environment',
            filepaths={'environment': '/some/gallery/test/environment.py'},
            metadata={'difficulty': 3},
        )
        cls.policy = Policy.objects.create(
            name='Test policy',
            filepaths={'policy': '/some/gallery/test/policy.py',
                       'helper': '/some/gallery/test/helper.py'},
            checkpoint_interval=-2,
        )
        cls.human = Agent.objects.create(
            role='agent_0', name='Human', participant=True,
            keyboard_inputs={'ArrowLeft': 1, 'ArrowRight': 2, 'default': 0},
            keyboard_input_display={'ArrowLeft': {'symbol': '←', 'label': 'Left'}},
        )
        cls.robot = Agent.objects.create(role='agent_1', name='Robot', policy=cls.policy)
        cls.experiment = Experiment.objects.create(
            name='Test experiment', link='test', environment=cls.environment,
            short_description='Short', long_description='First line\nSecond line\n',
            redirect_url='https://example.com/done', target_fps=10.0,
            dependencies=['gymnasium[classic_control]'],
        )
        cls.experiment.agents.add(cls.human, cls.robot)

    def test_top_level_keys_follow_gallery_layout(self):
        """The config has the same top-level keys, in the same order, as Gallery configs."""
        config = export_experiment(self.experiment)

        self.assertEqual(
            list(config),
            ['use_case', 'dependencies', 'environment', 'policy', 'agents', 'experiment'],
        )

    def test_dependencies_are_exported_at_top_level_only(self):
        """Dependencies are written as a top-level list, not inside the experiment block."""
        config = export_experiment(self.experiment)

        self.assertEqual(config['dependencies'], ['gymnasium[classic_control]'])
        self.assertNotIn('dependencies', config['experiment'])

    def test_filepaths_are_exported_as_bare_filenames(self):
        """Stored paths are reduced to filenames, as files sit next to config.yaml."""
        config = export_experiment(self.experiment)

        self.assertEqual(config['environment']['filepaths'], {'environment': 'environment.py'})
        self.assertEqual(config['policy']['filepaths'],
                         {'policy': 'policy.py', 'helper': 'helper.py'})

    def test_agents_reference_their_policy_by_name(self):
        """Agents are exported in creation order and refer to their policy by name."""
        agents = export_experiment(self.experiment)['agents']

        self.assertEqual([agent['role'] for agent in agents], ['agent_0', 'agent_1'])
        self.assertIsNone(agents[0]['policy'])
        self.assertEqual(agents[1]['policy'], 'Test policy')

    def test_policy_block_is_omitted_without_policy(self):
        """An experiment whose agents have no policy gets no policy block."""
        self.experiment.agents.remove(self.robot)

        config = export_experiment(self.experiment)

        self.assertNotIn('policy', config)

    def test_experiment_with_several_policies_is_refused(self):
        """config.yaml holds a single policy, so experiments using several are refused."""
        self.human.policy = Policy.objects.create(name='Other policy',
                                                  filepaths={'policy': '/other/policy_b.py'})
        self.human.save()

        with self.assertRaises(ExportError):
            export_experiment(self.experiment)

    def test_files_sharing_a_filename_are_refused(self):
        """Files that would share a filename next to config.yaml are refused."""
        self.environment.filepaths = {'environment': '/env_code/environment.py',
                                      'utils': '/env_code/utils.py'}
        self.environment.save()
        self.policy.filepaths = {'policy': '/policy_code/policy.py',
                                 'utils': '/policy_code/utils.py'}
        self.policy.save()

        with self.assertRaises(ExportError):
            export_experiment(self.experiment)

    def test_yaml_text_loads_back_to_the_exported_config(self):
        """The downloaded YAML text contains exactly the exported config."""
        text = export_experiment_yaml(self.experiment)

        self.assertEqual(yaml.safe_load(text), export_experiment(self.experiment))

    def test_yaml_text_uses_gallery_formatting(self):
        """Non-ASCII characters are escaped and multi-line texts use readable blocks."""
        text = export_experiment_yaml(self.experiment)

        self.assertIn('symbol: "\\u2190"', text)
        self.assertIn('long_description: |', text)

    def test_exported_config_reinstalls_identically(self):
        """Installing the exported YAML on an empty database recreates the same experiment."""
        exported = export_experiment(self.experiment)
        text = export_experiment_yaml(self.experiment)
        self.experiment.delete()
        Agent.objects.all().delete()
        Policy.objects.all().delete()
        Environment.objects.all().delete()

        setup_database(yaml.safe_load(text), verbosity=0)

        reinstalled = Experiment.objects.get(link=self.experiment.link)
        self.assertEqual(export_experiment(reinstalled), exported)

    def test_config_without_dependencies_installs_with_empty_list(self):
        """A config whose dependencies are missing or empty installs with no dependencies."""
        config = export_experiment(self.experiment)
        del config['dependencies']
        cases = {'missing key': config, 'empty value': {**config, 'dependencies': None}}

        for case, case_config in cases.items():
            with self.subTest(case):
                setup_database(case_config, verbosity=0)

                self.experiment.refresh_from_db()
                self.assertEqual(self.experiment.dependencies, [])

    def test_dependencies_given_as_a_string_are_rejected(self):
        """A single string instead of a list of requirements is refused."""
        self.experiment.dependencies = 'gymnasium'

        with self.assertRaises(ValidationError) as raised:
            self.experiment.save()

        self.assertIn('dependencies', raised.exception.message_dict)


class ExportAsYamlActionTests(TestCase):
    """Tests for the admin action that downloads an experiment as config.yaml."""

    @classmethod
    def setUpTestData(cls):
        """Create two experiments and an admin user."""
        environment = Environment.objects.create(name='Environment',
                                                 filepaths={'environment': '/x/environment.py'})
        cls.first = Experiment.objects.create(name='First', link='first', environment=environment,
                                              short_description='s', long_description='l')
        cls.second = Experiment.objects.create(name='Second', link='second',
                                               environment=environment,
                                               short_description='s', long_description='l')
        cls.admin_user = User.objects.create_superuser('admin', password='password')

    def setUp(self):
        """Log in as the admin user."""
        self.client.force_login(self.admin_user)

    def run_export_action(self, *experiments):
        """Submit the export action for the given experiments, as the admin list page does."""
        return self.client.post(reverse('admin:experiment_experiment_changelist'), {
            'action': 'export_as_yaml',
            '_selected_action': [experiment.pk for experiment in experiments],
        })

    def test_exporting_one_experiment_downloads_its_yaml(self):
        """Exporting a single experiment downloads its config.yaml."""
        response = self.run_export_action(self.first)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Disposition'], 'attachment; filename="first.yaml"')
        self.assertEqual(yaml.safe_load(response.content)['experiment']['link'], 'first')

    def test_exporting_several_experiments_shows_an_error(self):
        """Selecting more than one experiment shows an error instead of downloading."""
        response = self.run_export_action(self.first, self.second)

        self.assertNotIn('Content-Disposition', response)
        shown = [message.level for message in get_messages(response.wsgi_request)]
        self.assertEqual(shown, [messages.ERROR])

    def test_staff_without_view_permission_cannot_export(self):
        """Staff users who may not view experiments cannot download them."""
        staff_user = User.objects.create_user('staff', password='password', is_staff=True)
        self.client.force_login(staff_user)

        response = self.run_export_action(self.first)

        self.assertEqual(response.status_code, 403)

    def test_experiment_refused_by_the_exporter_shows_an_error(self):
        """An experiment the exporter refuses shows the reason instead of crashing."""
        for name in ('Policy A', 'Policy B'):
            policy = Policy.objects.create(name=name, filepaths={'policy': f'/{name}/policy.py'})
            self.first.agents.add(Agent.objects.create(role=name, name=name, policy=policy))

        response = self.run_export_action(self.first)

        self.assertNotIn('Content-Disposition', response)
        shown = [message.level for message in get_messages(response.wsgi_request)]
        self.assertEqual(shown, [messages.ERROR])


@skipUnless(GALLERY_DIR.exists(), 'Clone the SHARPIE_Gallery repository next to this '
                                  'repository (or set SHARPIE_GALLERY_DIR) to run these tests.')
class GalleryRoundTripTests(TestCase):
    """
    Install, export, reinstall, and re-export every use case of a gallery clone.

    This guards the installer <-> exporter round trip against the real gallery
    configs, like the release checklist's "check gallery installs works" step.
    Each use case's database is rebuilt from scratch twice: once from its
    original config, once from that config's export, and both exports must be
    identical. Strict equality with the original file is not asserted, because
    the exporter writes every editable model field (e.g. null metadata) while
    hand-written configs omit them. Top-level keys the installer never persists
    (e.g. python_version, installation_notes) are therefore not round-tripped.

    The test bypasses steps 1 and 2 of sharpie-install (pip installation and
    code validation) and exercises only the config.yaml <-> database path, so
    no use case needs its runtime dependencies installed. The filepaths stay
    bare in both passes because the conversion to absolute paths is bypassed.
    """

    def discover_use_cases(self):
        """Every gallery folder with a config.yaml, like sharpie-install --list does."""
        return sorted(path.name for path in GALLERY_DIR.iterdir()
                      if (path / 'config.yaml').exists())

    def wipe(self):
        """Remove gallery-linked rows so every use case installs in isolation."""
        Experiment.objects.all().delete()
        Agent.objects.all().delete()
        Policy.objects.all().delete()
        Environment.objects.all().delete()

    def test_discovery_matches_sharpie_install(self):
        """Use cases are discovered exactly like sharpie-install --list discovers them."""
        self.assertEqual(self.discover_use_cases(), list_use_cases(GALLERY_DIR))

    def test_every_use_case_round_trips_identically(self):
        """Reinstalling an export recreates an experiment that exports identically."""
        use_cases = self.discover_use_cases()
        self.assertTrue(use_cases, f'No use cases found in {GALLERY_DIR}')

        for use_case in use_cases:
            with self.subTest(use_case=use_case):
                # First pass: install the original gallery config.
                self.wipe()
                config = load_config(use_case, GALLERY_DIR)
                setup_database(config, verbosity=0)
                experiment = Experiment.objects.get(link=config['experiment']['link'])
                exported = export_experiment(experiment)
                # The YAML text is generated before the wipe: agents are a
                # many-to-many relation and would read back as empty after it.
                text = export_experiment_yaml(experiment)

                # Second pass: install the export on a clean database.
                self.wipe()
                setup_database(yaml.safe_load(text), verbosity=0)
                reinstalled = Experiment.objects.get(link=experiment.link)

                self.assertEqual(export_experiment(reinstalled), exported)
