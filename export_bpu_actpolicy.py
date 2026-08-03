from pathlib import Path
import runpy


runpy.run_path(
    str(
        Path(__file__).resolve().parent
        / "models"
        / "act"
        / "export_bpu_actpolicy.py"
    ),
    run_name="__main__",
)
