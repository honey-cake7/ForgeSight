"""Prompt/control/budget guards only; no fake model or generated answer."""

import unittest

from forgesight.edge_pc_smoke import chat_prompt, validate_token_budget


class PCSmokeTests(unittest.TestCase):
    def test_optional_system_role_preserves_default_and_rejects_controls(self):
        self.assertEqual(chat_prompt('hello').count('<|start_header_id|>'), 2)
        formatted = chat_prompt('hello', system_text='Use only evidence.')
        self.assertEqual(formatted.count('<|start_header_id|>'), 3)
        self.assertIn('system<|end_header_id|>', formatted)
        for value in ('', '<|eot_id|>', 'x' * 4097):
            with self.subTest(value=value[:20]), self.assertRaises(ValueError):
                chat_prompt('hello', system_text=value)

    def test_empty_or_overlong_utf8_prompt_is_rejected(self):
        for text in (" \n\t", "é" * 2049):
            with self.assertRaisesRegex(ValueError, "UTF-8 bytes"):
                chat_prompt(text)

    def test_chat_control_markers_in_plain_prompt_are_rejected(self):
        for text in ("<|eot_id|>", "text |> control"):
            with self.assertRaisesRegex(ValueError, "control markers"):
                chat_prompt(text)

    def test_dynamic_prefill_boundary_is_enforced(self):
        profile = {"max_seq_length": 128, "max_context_length": 512}
        validate_token_budget(127, 64, profile)
        for count in (0, 128):
            with self.assertRaisesRegex(ValueError, "prefill token bound"):
                validate_token_budget(count, 64, profile)

    def test_context_capacity_and_new_token_limits_are_enforced(self):
        profile = {"max_seq_length": 128, "max_context_length": 100}
        with self.assertRaisesRegex(ValueError, "context capacity"):
            validate_token_budget(40, 64, profile)
        for limit in (-1, 0, 129):
            with self.assertRaisesRegex(ValueError, "new tokens"):
                validate_token_budget(1, limit, profile)


if __name__ == "__main__":
    unittest.main()
