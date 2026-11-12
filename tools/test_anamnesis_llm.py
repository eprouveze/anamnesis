#!/usr/bin/env python3
"""Tests for anamnesis_llm: opt-in provider, timeout validation, scrubbed child env.
No network and no API key."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import anamnesis_llm as llm  # noqa: E402


class TestLLM(unittest.TestCase):
    def setUp(self):
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        for k in ("ANAMNESIS_LLM_PROVIDER", "ANAMNESIS_LLM_COMMAND", "ANAMNESIS_LLM_TIMEOUT_S", "ANAMNESIS_LLM_MODEL"):
            os.environ.pop(k, None)

    def tearDown(self):
        self._env.stop()

    def test_no_provider_refuses_and_names_what_to_set(self):
        with self.assertRaises(llm.LLMUnavailable) as cm:
            llm.complete("hi")
        self.assertIn("ANAMNESIS_LLM_PROVIDER", str(cm.exception))

    def test_default_model(self):
        self.assertEqual(llm.default_model(), "gemini-3.8-flash")

    def test_bad_timeout_is_llm_unavailable(self):
        os.environ["ANAMNESIS_LLM_PROVIDER"] = "command"
        os.environ["ANAMNESIS_LLM_COMMAND"] = sys.executable + " -c pass"
        for bad in ("abc", "0", "-3", "nan", "inf"):
            os.environ["ANAMNESIS_LLM_TIMEOUT_S"] = bad
            with self.assertRaises(llm.LLMUnavailable, msg=bad):
                llm.complete("hi")

    def test_command_child_env_is_scrubbed(self):
        os.environ["ANAMNESIS_LLM_PROVIDER"] = "command"
        os.environ["FAKE_SERVICE_API_KEY"] = "k"
        os.environ["FAKE_ACCESS_TOKEN"] = "t"
        os.environ["FAKE_CLIENT_SECRET"] = "s"
        os.environ["FAKE_PLAIN_SETTING"] = "keep"
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "echo_env.py"
            script.write_text(
                "import os,sys\nsys.stdin.read()\n"
                "print(sorted(k for k in os.environ if k.startswith('FAKE_')))\n")
            os.environ["ANAMNESIS_LLM_COMMAND"] = f"{sys.executable} {script}"
            out = llm.complete("prompt")
        self.assertEqual(out, "['FAKE_PLAIN_SETTING']")

    def test_unknown_provider(self):
        os.environ["ANAMNESIS_LLM_PROVIDER"] = "other"
        with self.assertRaises(llm.LLMUnavailable):
            llm.complete("hi")




class ScrubPassthroughTests(unittest.TestCase):
    def test_broad_markers_and_explicit_passthrough(self):
        import anamnesis_llm
        env = {"GEMINI_API_KEY_BACKUP": "x", "AWS_SECRET_ACCESS_KEY": "x", "DB_PASSWORD": "x",
               "GOOGLE_APPLICATION_CREDENTIALS": "x", "ANAMNESIS_OAUTH_JWT_KEY": "x",
               "CHILD_SERVICE_KEY": "keep", "PATH": "/usr/bin"}
        with mock.patch.dict(os.environ, env, clear=True):
            os.environ["ANAMNESIS_LLM_COMMAND_ENV_PASS"] = "CHILD_SERVICE_KEY"
            out = anamnesis_llm._scrubbed_env()
        self.assertEqual(out.get("CHILD_SERVICE_KEY"), "keep")
        self.assertEqual(out.get("PATH"), "/usr/bin")
        for k in ("GEMINI_API_KEY_BACKUP", "AWS_SECRET_ACCESS_KEY", "DB_PASSWORD",
                  "GOOGLE_APPLICATION_CREDENTIALS", "ANAMNESIS_OAUTH_JWT_KEY"):
            self.assertNotIn(k, out)


if __name__ == "__main__":
    unittest.main()
