"""UI smoke tests via Streamlit's AppTest on a SYNTHETIC database, with pandas BLOCKED to emulate the Windows
Application Control crash ('DLL load failed while importing ccalendar')."""
import json
import os
import sys

import pytest

from pathlib import Path
from tests.fixtures import RAW

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    d = tmp_path_factory.mktemp("ui")
    os.environ["RECIPE_DB_PATH"] = str(d / "r.db")
    import importlib
    from core import config
    importlib.reload(config)
    from scripts.build_database import build
    names = ["a.json", "b.json", "c.json"]
    for i, n in enumerate(names):
        (d / n).write_text(json.dumps({f"h{i}{k}": {"title": t, "ingredients": x, "instructions": s} for k, (t, x, s) in enumerate(RAW[i::3])}))
    build(d, d / "r.db", expected=names)
    saved = sys.modules.get("pandas", "absent")
    sys.modules["pandas"] = None            # any `import pandas` now raises ImportError
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=90).run()
    yield at
    if saved == "absent":
        sys.modules.pop("pandas", None)
    else:
        sys.modules["pandas"] = saved


def test_loads_and_chat_input_present(app):
    assert not app.exception
    assert len(app.chat_input) == 1


def test_all_task_paths_render_without_pandas(app):
    for q in ["Find a penne pasta recipe", "I cannot have dairy", "I have eggs, onions, tomatoes and rice. No peanuts. At most 6 steps.",
              "Exclude eggs too", "How many recipes contain both garlic and onion?", "Among recipes containing tomato, which five have the fewest instruction steps?",
              "Remove eggs from the banana bread recipe", "What can I substitute for cream?", "I have noodles. At most 0 steps.", "show ingredients of result 1"]:
        app.chat_input[0].set_value(q).run()
        assert not app.exception, (q, [str(x.value) for x in app.exception])
        assert len(app.chat_input) == 1                                   # input remains after every response
    assert "pandas" not in [m for m, v in sys.modules.items() if v is not None and m == "pandas"]


def test_example_buttons_and_forms_work(app):
    ex = [b for b in app.button if b.label.startswith("Find a quick chicken")]
    assert ex
    ex[0].click().run()
    assert not app.exception
    assert any("quick chicken" in m["content"] for m in app.session_state.messages)
    app.button(key="reset_top").click().run()
    assert app.session_state.messages == []


def test_no_pandas_backed_widgets_used_in_source():
    import pathlib, re
    root = pathlib.Path(APP).parent
    for f in list((root / "ui").glob("*.py")) + [root / "app.py"]:
        src = f.read_text(encoding="utf-8")
        assert not re.search(r"st\.(dataframe|table|data_editor|line_chart|bar_chart)\(", src), f
        assert not re.search(r"^\s*(import|from)\s+pandas", src, re.M) and "st.write(" not in src, f
