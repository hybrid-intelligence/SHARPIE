from django.test import TestCase


class ExperimentTestCase(TestCase):
    def setUp(self):
        pass

    def test_config(self):
        """Test configuration can be saved and loaded."""
        self.assertEqual(1, 0)

    def test_run(self):
        """Test experiment can be run and is saved to the database."""
        self.assertEqual(1, 0)