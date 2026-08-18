import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"


class AppTests(unittest.TestCase):
    def test_example_mode_is_read_only(self):
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, "Anticiper la consommation d'un bâtiment")
        self.assertGreaterEqual(len(app.metric), 6)
        self.assertEqual(app.segmented_control[0].value, "Explorer des exemples")
        self.assertEqual(len(app.button), 0)
        self.assertIn("Consommation observée", [metric.label for metric in app.metric])
        surface = next(metric for metric in app.metric if metric.label == "Surface")
        self.assertIn("m²", surface.value)

    def test_simulation_mode_is_editable_without_observed_value(self):
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        app.segmented_control[0].set_value("Simuler un bâtiment").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.button[0].label, "Estimer la consommation")
        self.assertEqual(app.number_input[0].label, "Surface du bâtiment (m²)")
        labels = [metric.label for metric in app.metric]
        self.assertNotIn("Consommation observée", labels)
        self.assertIn("Médiane du jeu de test", labels)


if __name__ == "__main__":
    unittest.main()
