from __future__ import annotations

from src.data_pipeline import run_full_pipeline


if __name__ == "__main__":
    result = run_full_pipeline()
    print("Pipeline completed successfully.")
    print(result)
