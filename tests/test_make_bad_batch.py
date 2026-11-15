from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import pandera.pandas as pa

from src.contract import FEATURE_COLUMNS, schema
from src.make_bad_batch import make_batches

ROOT = Path(__file__).resolve().parents[1]


class BatchDemoTests(unittest.TestCase):
    def test_import_has_no_filesystem_side_effects(self):
        with tempfile.TemporaryDirectory() as temp:
            result = subprocess.run(
                [sys.executable, "-c",
                 f"import sys; sys.path.insert(0, {str(ROOT)!r}); import src.make_bad_batch"],
                cwd=temp, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_generated_batches_demonstrate_quality_not_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = root / "holdout.parquet", root / "incoming"
            data = pd.DataFrame({
                "SeriousDlqin2yrs": np.tile([0, 1], 300),
                **{c: np.arange(600, dtype=float) + 1 for c in FEATURE_COLUMNS},
            })
            for column, definition in schema.columns.items():
                if str(definition.dtype) == "int64":
                    data[column] = data[column].astype(int)
            data["age"] = 45
            data.to_parquet(source)
            original = source.read_bytes()
            make_batches(source, output)
            good = schema.validate(pd.read_parquet(output / "good_batch.parquet"), lazy=True)
            self.assertEqual(len(good), 500)
            bad = pd.read_parquet(output / "bad_batch.parquet")
            self.assertEqual(len(bad), 505)
            with self.assertRaises(pa.errors.SchemaErrors):
                schema.validate(bad, lazy=True)
            self.assertEqual(source.read_bytes(), original)
            data.head(100).to_parquet(source)
            with self.assertRaisesRegex(ValueError, "500"):
                make_batches(source, root / "too_small")
            self.assertFalse((root / "too_small").exists())
