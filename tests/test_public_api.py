import subprocess
import sys
import unittest

import evojax
from evojax.algo import NEAlgorithm, PGPE
from evojax.policy import HyperNetwork, ParameterAdapter


class PublicApiTests(unittest.TestCase):
    def test_core_imports_have_no_legacy_dependencies(self):
        program = """
import sys
import evojax
import evojax.algo
import evojax.policy
import evojax.algo.cultural
for name in ("torch", "torchvision", "matplotlib", "cma", "brax", "evosax"):
    assert name not in sys.modules, name
assert "evojax.trainer" not in sys.modules
"""
        subprocess.run([sys.executable, "-c", program], check=True, capture_output=True)

    def test_public_types_and_version(self):
        self.assertTrue(issubclass(PGPE, NEAlgorithm))
        self.assertTrue(evojax.__version__)
        self.assertTrue(callable(HyperNetwork))
        self.assertTrue(callable(ParameterAdapter))
