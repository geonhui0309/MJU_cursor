"""
Streamlit Cloud / 루트 실행용 진입점.
실제 앱 코드는 survey-insight-app/app.py 에 있습니다.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent / "survey-insight-app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

_app_file = APP_DIR / "app.py"
_spec = importlib.util.spec_from_file_location("survey_insight_main", _app_file)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"앱 파일을 불러올 수 없습니다: {_app_file}")
_module = importlib.util.module_from_spec(_spec)
sys.modules["survey_insight_main"] = _module
_spec.loader.exec_module(_module)
