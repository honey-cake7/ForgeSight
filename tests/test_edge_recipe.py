"""Recipe tests exercise metadata/guards, never export or simulate a model."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from forgesight.edge_recipe import create_recipe, export_config, validate_export_config


class RecipeTests(unittest.TestCase):
    def setUp(self):
        self.profile = json.loads(Path("configs/edge_llama.json").read_text())
        self.schema = json.loads(Path("configs/edge_upstream_schema.json").read_text())

    def test_export_fields_match_pinned_schema(self):
        config = export_config(self.profile, Path("weights"), Path("output"))
        validate_export_config(config, self.schema)
        self.assertEqual(config["quantization"]["qmode"], "8da4w")
        self.assertTrue(config["backend"]["xnnpack"]["enabled"])
        self.assertEqual(config["export"]["output_name"], "llama3_2_1b_xnnpack")

    def test_unknown_upstream_key_is_rejected(self):
        config = export_config(self.profile, Path("weights"), Path("output"))
        config["export"]["imaginary_flag"] = True
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            validate_export_config(config, self.schema)

    def test_context_smaller_than_sequence_is_rejected(self):
        profile = deepcopy(self.profile)
        profile["max_context_length"] = 64
        with self.assertRaisesRegex(ValueError, "sequence/context"):
            validate_export_config(export_config(profile, Path("w"), Path("o")), self.schema)

    def test_unreviewed_backend_is_not_substituted(self):
        profile = deepcopy(self.profile)
        profile["backend"] = "ollama"
        with self.assertRaisesRegex(ValueError, "Unreviewed backend"):
            export_config(profile, Path("w"), Path("o"))

    def test_source_revision_mismatch_is_rejected(self):
        profile = deepcopy(self.profile)
        profile["executorch_commit"] = "0" * 40
        with self.assertRaisesRegex(ValueError, "different source commits"):
            create_recipe(profile, self.schema, Path("c"), Path("w"), Path("unused"))

    def test_missing_files_remain_blocked_and_no_export_is_created(self):
        with tempfile.TemporaryDirectory(prefix="forgesight-recipe-") as directory:
            base = Path(directory)
            output = base / "recipe with spaces"
            recipe = create_recipe(self.profile, self.schema, base / "checkout with spaces",
                                   base / "weights", output)
            self.assertEqual(recipe["status"], "PREPARATION_ONLY_NOT_RUN")
            self.assertFalse(recipe["ready_to_claim_export"])
            self.assertIsNone(recipe["generated_text"])
            self.assertEqual(len(recipe["missing_model_files"]), 5)
            self.assertEqual({path.name for path in output.iterdir()}, {"recipe.json", "export_config.yaml"})
            command = recipe["export_command_not_run"]
            self.assertIsInstance(command["argv"], list)
            self.assertEqual(command["argv"][-1], str((output / "export_config.yaml").resolve()))
            with self.assertRaises(FileExistsError):
                create_recipe(self.profile, self.schema, base / "checkout", base / "weights", output)

    def test_official_cpu_build_matches_release_but_other_versions_do_not(self):
        with tempfile.TemporaryDirectory(prefix="forgesight-version-") as directory:
            base = Path(directory)
            for index, version in enumerate(("1.5.1+cpu", "1.5.1rc1", "1.5.0+cpu")):
                with patch("forgesight.edge_recipe.importlib.metadata.version", return_value=version):
                    recipe = create_recipe(self.profile, self.schema, base / "checkout",
                                           base / "weights", base / str(index))
                self.assertEqual(recipe["installed_executorch"], version)
                missing_environment = any("not installed here" in item for item in recipe["blockers"])
                self.assertEqual(missing_environment, index != 0)
                self.assertFalse(recipe["ready_to_claim_export"])


if __name__ == "__main__":
    unittest.main()
