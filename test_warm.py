import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from finflux.modules.insight_engine.financial_models import ProductionExpertModule

try:
    expert = ProductionExpertModule()
    print("Pre-warming...")
    expert.warm()
    print("Warm-up complete.")
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
