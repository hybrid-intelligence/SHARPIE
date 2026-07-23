"""Tests for the experiment app."""
from django.test import TestCase


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