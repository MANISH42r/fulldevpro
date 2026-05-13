"""Default entry for Streamlit Community Cloud — runs CareerLens from app1.py."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parent / "app1.py"), run_name="__main__")
