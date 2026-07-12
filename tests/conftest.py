import sys
import types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if "anomalib.data" not in sys.modules:
    try:
        import anomalib.data  # noqa: F401  -- deja installe, on garde le vrai
    except ImportError:
        fake_anomalib_data = types.ModuleType("anomalib.data")

        class _FakePredictDataset:
            def __init__(self, *args, **kwargs):
                pass

        fake_anomalib_data.PredictDataset = _FakePredictDataset
        sys.modules.setdefault("anomalib", types.ModuleType("anomalib"))
        sys.modules["anomalib.data"] = fake_anomalib_data