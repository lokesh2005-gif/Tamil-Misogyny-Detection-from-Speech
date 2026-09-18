"""
Unit Tests for Centralized APIKeyManager.
Verifies multi-key detection, round-robin rotation, failover on rate-limiting,
bounded retry safety, and zero key leakage.
"""

import os
import unittest
from unittest.mock import patch, MagicMock
from src.key_manager import APIKeyManager, is_rate_limit_or_quota_error


class TestAPIKeyManager(unittest.TestCase):

    def setUp(self):
        # Create an isolated manager with a short 2-second cooldown for test speed
        self.km = APIKeyManager(cooldown_seconds=2.0)

    def test_multi_key_detection_and_filtering(self):
        """Verify indexed keys, generic keys, and empty/placeholder filtering."""
        mock_env = {
            "GEMINI_API_KEY_1": "test_gemini_key_alpha",
            "GEMINI_API_KEY_2": "test_gemini_key_beta",
            "GEMINI_API_KEY_3": "   ",  # whitespace only
            "GEMINI_API_KEY_4": "",     # empty
            "GEMINI_API_KEY_5": "your_api_key_here",  # placeholder
            "API_KEY_1": "test_generic_key_gamma",
            "API_KEY_2": "test_gemini_key_alpha",  # duplicate, must be deduplicated
        }
        with patch.dict(os.environ, mock_env, clear=True):
            self.km.reload_keys(load_env_file=False)
            count = self.km.get_key_count("gemini")
            # Should have 3 unique valid keys: alpha, beta, gamma
            self.assertEqual(count, 3)

    def test_round_robin_sequence(self):
        """Test KEY 1 -> KEY 2 -> KEY 3 -> KEY 1 round-robin rotation."""
        mock_env = {
            "API_KEY_1": "key_one",
            "API_KEY_2": "key_two",
            "API_KEY_3": "key_three",
        }
        with patch.dict(os.environ, mock_env, clear=True):
            self.km.reload_keys(load_env_file=False)

            k1, a1 = self.km.get_next_key("gemini")
            k2, a2 = self.km.get_next_key("gemini")
            k3, a3 = self.km.get_next_key("gemini")
            k4, a4 = self.km.get_next_key("gemini")

            self.assertEqual(k1, "key_one")
            self.assertEqual(k2, "key_two")
            self.assertEqual(k3, "key_three")
            self.assertEqual(k4, "key_one")  # Wraps around
            self.assertEqual(a1, "Key-1")
            self.assertEqual(a2, "Key-2")
            self.assertEqual(a3, "Key-3")
            self.assertEqual(a4, "Key-1")

    def test_rate_limit_temporary_skip(self):
        """When a key is marked rate-limited, round-robin temporarily skips it."""
        mock_env = {
            "API_KEY_1": "key_alpha",
            "API_KEY_2": "key_beta",
        }
        with patch.dict(os.environ, mock_env, clear=True):
            self.km.reload_keys(load_env_file=False)

            # Mark key_alpha as rate-limited
            self.km.mark_rate_limited("gemini", "key_alpha", "ResourceExhausted")

            # Subsequent requests should skip key_alpha and use key_beta
            k1, _ = self.km.get_next_key("gemini")
            self.assertEqual(k1, "key_beta")

            k2, _ = self.km.get_next_key("gemini")
            self.assertEqual(k2, "key_beta")

    def test_automatic_failover_on_quota_error(self):
        """Verify execute_with_failover switches key automatically on 429 / quota."""
        mock_env = {
            "API_KEY_1": "failing_quota_key",
            "API_KEY_2": "working_backup_key",
        }
        with patch.dict(os.environ, mock_env, clear=True):
            self.km.reload_keys(load_env_file=False)

            call_log = []

            def mock_api_call(key: str):
                call_log.append(key)
                if key == "failing_quota_key":
                    raise Exception("429 RESOURCE_EXHAUSTED: Quota exceeded for model")
                return "SUCCESSFUL_RESPONSE"

            res = self.km.execute_with_failover("gemini", mock_api_call)

            self.assertEqual(res, "SUCCESSFUL_RESPONSE")
            self.assertEqual(call_log, ["failing_quota_key", "working_backup_key"])

    def test_bounded_retries_does_not_loop_indefinitely(self):
        """When all keys fail with quota error, manager halts cleanly after total_keys attempts."""
        mock_env = {
            "API_KEY_1": "fail_1",
            "API_KEY_2": "fail_2",
            "API_KEY_3": "fail_3",
        }
        with patch.dict(os.environ, mock_env, clear=True):
            self.km.reload_keys(load_env_file=False)

            attempts = []

            def always_fails(key: str):
                attempts.append(key)
                raise Exception("429 Too Many Requests")

            with self.assertRaises(RuntimeError) as ctx:
                self.km.execute_with_failover("gemini", always_fails)

            # Must have tried exactly 3 times (once per key)
            self.assertEqual(len(attempts), 3)
            self.assertIn("All 3 configured API keys for 'gemini' failed", str(ctx.exception))

    def test_zero_credential_leakage_in_errors_and_telemetry(self):
        """Verify raw keys are never leaked in error messages or pool status."""
        secret = "super_secret_token_xyz987"
        mock_env = {
            "API_KEY_1": secret,
        }
        with patch.dict(os.environ, mock_env, clear=True):
            self.km.reload_keys(load_env_file=False)

            # 1. Error sanitization
            err = self.km.sanitize_text(f"Connection refused at https://api.google.com with key={secret}")
            self.assertNotIn(secret, err)
            self.assertIn("[REDACTED_API_KEY]", err)

            # 2. Pool status must not contain secret
            status = self.km.get_pool_status("gemini")
            status_str = str(status)
            self.assertNotIn(secret, status_str)
            self.assertEqual(status["total_keys"], 1)
            self.assertEqual(status["active_keys"], 1)

    def test_quota_error_cue_detection(self):
        """Verify rate limit detection heuristics."""
        self.assertTrue(is_rate_limit_or_quota_error(Exception("Error code 429: Too Many Requests")))
        self.assertTrue(is_rate_limit_or_quota_error(Exception("RESOURCE_EXHAUSTED: quota reached")))
        self.assertTrue(is_rate_limit_or_quota_error(Exception("insufficient_quota: billing limit")))
        self.assertFalse(is_rate_limit_or_quota_error(ValueError("Invalid audio file format")))


if __name__ == "__main__":
    unittest.main()
