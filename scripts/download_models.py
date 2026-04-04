"""FinFlux V4.2 Model Pre-Cache Script (Hardened).
Run this to ensure all 7 local AI models are fully cached on disk.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from transformers import pipeline
except ImportError:
    print("Error: transformers not installed.")
    exit(1)

# Import directly from config so script and module always stay in sync
try:
    from src.finflux import config
except ImportError:
    print("Error: Could not import finflux.config. Run from project root.")
    exit(1)

MODELS = [
    (config.HF_LANG_DETECT,  "text-classification"),
    (config.HF_ZERO_SHOT,    "zero-shot-classification"),
    (config.HF_FINBERT,      "sentiment-analysis"),
    (config.HF_NER_GENERAL,  "ner"),
    (config.HF_INDIC_NER,    "ner"),
    (config.HF_INDIC_STT,    "automatic-speech-recognition"),
]

def download_transformers():
    print("\n--- Downloading Transformer Models ---")
    failed = []
    for model_id, task in MODELS:
        try:
            print(f"  Checking [{task}] {model_id} ...")
            pipeline(task, model=model_id)
            print(f"  ✓ Cached: {model_id}")
        except Exception as e:
            print(f"  ✗ FAILED: {model_id}\n    Reason: {e}")
            failed.append(model_id)
    return failed

def download_gliner():
    print("\n--- Downloading GLiNER Specialist ---")
    try:
        from gliner import GLiNER
        print(f"  Checking {config.HF_NER_FINANCIAL} ...")
        GLiNER.from_pretrained(config.HF_NER_FINANCIAL)
        print(f"  ✓ Cached: {config.HF_NER_FINANCIAL}")
        return []
    except ImportError:
        print("  GLiNER package not installed — skipping. Install with: pip install gliner")
        return []
    except Exception as e:
        print(f"  ✗ FAILED: {config.HF_NER_FINANCIAL}\n    Reason: {e}")
        return [config.HF_NER_FINANCIAL]

if __name__ == "__main__":
    print("=== FinFlux V4.2 Model Pre-Cache Script ===")

    failed = []
    failed += download_transformers()
    failed += download_gliner()

    print("\n=== Pre-Cache Complete ===")
    if failed:
        print(f"\n⚠ {len(failed)} model(s) failed to download:")
        for m in failed:
            print(f"   - {m}")
        print("\nFix the errors above before starting the server.")
        sys.exit(1)  # Non-zero exit so CI/CD pipelines catch failures
    else:
        print("\n✓ All models cached. Start the server with:")
        print("  python -m uvicorn api.server:app --reload --port 8000")