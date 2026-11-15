#!/usr/bin/env python3
"""Unit tests for anamnesis_dict.py (stdlib only; temp dictionaries).

    python -m unittest discover -s tools -p 'test_*.py' -v
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import anamnesis_dict as ad  # noqa: E402

ENV_KEYS = ("ANAMNESIS_DICT_PATH", "ANAMNESIS_DATA_DIR", "ANAMNESIS_HOME", "ANAMNESIS_DB",
            "ANAMNESIS_REPO_ROOT", "ANAMNESIS_PATH_ROOTS")


class DictCase(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in ENV_KEYS}
        for k in ENV_KEYS:
            os.environ.pop(k, None)
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        ad.reset_cache()

    def tearDown(self):
        self._tmp.cleanup()
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        ad.reset_cache()

    def write(self, name, data):
        p = self.tmp / name
        p.write_text(json.dumps(data), encoding="utf-8")
        return p


class TestLoader(DictCase):
    def test_discovery_order_override_then_data_dir(self):
        a = self.write("a.json", {"projects": [{"name": "from-override"}]})
        (self.tmp / "data").mkdir()
        self.write("data/dictionary.json", {"projects": [{"name": "from-data-dir"}]})
        os.environ["ANAMNESIS_DATA_DIR"] = str(self.tmp / "data")
        self.assertEqual(ad.get_all_projects(), ["from-data-dir"])
        os.environ["ANAMNESIS_DICT_PATH"] = str(a)
        self.assertEqual(ad.get_all_projects(), ["from-override"])

    def test_cache_refreshes_when_the_file_changes(self):
        p = self.write("d.json", {"projects": [{"name": "one"}]})
        os.environ["ANAMNESIS_DICT_PATH"] = str(p)
        self.assertEqual(ad.get_all_projects(), ["one"])
        p.write_text(json.dumps({"projects": [{"name": "two"}]}), encoding="utf-8")
        os.utime(p, ns=(2_000_000_000_000_000_000, 2_000_000_000_000_000_000))
        self.assertEqual(ad.get_all_projects(), ["two"])

    def test_non_object_dictionary_is_rejected(self):
        p = self.tmp / "bad.json"
        p.write_text("[1, 2]", encoding="utf-8")
        os.environ["ANAMNESIS_DICT_PATH"] = str(p)
        with self.assertRaises(ValueError):
            ad.load_dictionary()

    def test_accessors(self):
        p = self.write("d.json", {
            "projects": [{"name": "a", "enabled": False}, {"name": "b"}],
            "store_types": {"decision": {"weight": 2, "resolve_target": True},
                            "memory": {"description": "no weight"}},
            "ontology": {"node_kinds": ["X"], "predicates": ["P"], "polysemy_blocklist": ["Patch"],
                         "spof_kinds": ["X"]},
            "registries": {"rules_files": ["R.md"]},
            "policies": [{"id": "p1"}, {"no_id": True}],
            "dev_root_candidates": ["work"], "path_roots": ["code/"],
        })
        os.environ["ANAMNESIS_DICT_PATH"] = str(p)
        os.environ["ANAMNESIS_PATH_ROOTS"] = os.pathsep.join(["extra", "work"])
        self.assertEqual(ad.get_all_projects(), ["b"])
        self.assertEqual(ad.get_all_projects(enabled_only=False), ["a", "b"])
        self.assertEqual(ad.get_store_weights(), {"decision": 2.0, "memory": 1.0})
        self.assertEqual(ad.get_store_types_with("resolve_target"), ("decision",))
        self.assertEqual(ad.get_store_types_with("nope", ("fallback",)), ("fallback",))
        self.assertEqual(ad.get_ontology_node_kinds(), {"X"})
        self.assertEqual(ad.get_ontology_predicates(), {"P"})
        self.assertEqual(ad.get_polysemy_blocklist(), {"patch"})
        self.assertEqual(ad.get_ontology_setting("spof_kinds"), ["X"])
        self.assertEqual(ad.get_registries(), {"rules_files": ["R.md"]})
        self.assertEqual([x["id"] for x in ad.get_policies()], ["p1"])
        self.assertEqual(ad.get_path_roots(), ["work", "code", "extra"])

    def test_resolve_project_path_uses_configured_roots(self):
        p = self.write("d.json", {"dev_root_candidates": ["work"]})
        os.environ["ANAMNESIS_DICT_PATH"] = str(p)
        (self.tmp / "home" / "work" / "demo").mkdir(parents=True)
        self.assertEqual(ad.resolve_project_path("demo", base_home=self.tmp / "home"),
                         self.tmp / "home" / "work" / "demo")
        self.assertIsNone(ad.resolve_project_path("missing", base_home=self.tmp / "home"))

    def test_locations_follow_environment(self):
        os.environ["ANAMNESIS_HOME"] = str(self.tmp / "h")
        self.assertEqual(ad.anamnesis_home(), self.tmp / "h")
        self.assertEqual(ad.db_path(), self.tmp / "h" / "memory.db")
        os.environ["ANAMNESIS_DB"] = str(self.tmp / "x.db")
        self.assertEqual(ad.db_path(), self.tmp / "x.db")
        os.environ["ANAMNESIS_DATA_DIR"] = str(self.tmp / "dd")
        self.assertEqual(ad.data_dir(), self.tmp / "dd")


if __name__ == "__main__":
    unittest.main()
