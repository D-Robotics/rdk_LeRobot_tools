from pathlib import Path
import runpy


runpy.run_path(
    str(Path(__file__).resolve().parent / "models" / "act" / "bpu_control_robot.py"),
    run_name="__main__",
)
