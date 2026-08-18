import unittest

from inference import build_scenario, load_artifacts, predict_scenario


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.without_score, cls.with_score, cls.metadata = load_artifacts()

    def test_three_demo_profiles_are_available(self):
        self.assertEqual(len(self.metadata["profiles"]), 3)

    def test_prediction_is_positive_and_deterministic(self):
        profile = self.metadata["profiles"][0]
        scenario = build_scenario(profile, {})
        first = predict_scenario(self.without_score, self.with_score, scenario)
        second = predict_scenario(self.without_score, self.with_score, scenario)
        self.assertAlmostEqual(
            first["with_energy_star_kbtu"], second["with_energy_star_kbtu"], places=5
        )
        self.assertAlmostEqual(
            first["without_energy_star_kbtu"], second["without_energy_star_kbtu"], places=5
        )
        self.assertGreater(first["with_energy_star_kbtu"], 0)
        self.assertGreater(first["without_energy_star_kbtu"], 0)

    def test_scenario_update_does_not_mutate_profile(self):
        profile = self.metadata["profiles"][0]
        initial_surface = profile["features"]["PropertyGFABuilding(s)"]
        scenario = build_scenario(profile, {"PropertyGFABuilding(s)": initial_surface * 2})
        self.assertEqual(profile["features"]["PropertyGFABuilding(s)"], initial_surface)
        self.assertEqual(scenario.iloc[0]["PropertyGFABuilding(s)"], initial_surface * 2)


if __name__ == "__main__":
    unittest.main()
