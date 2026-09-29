from pathlib import Path
import pytest
from crime_nlp.config import ROOT, PROCESSED

pytestmark = pytest.mark.skipif(not (PROCESSED / "dashboard.parquet").exists(), reason="Run the full pipeline before UI integration tests.")


def test_every_dashboard_page_and_narrative_analysis():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    assert not app.exception
    assert app.title[0].value == "From narratives to insight"
    for page in ["Case explorer", "Narrative lab", "Model evaluation", "Topics", "Pipeline & syllabus", "Learning guide"]:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, f"{page}: {app.exception}"
        if page == "Narrative lab":
            app.button[0].click().run()
            assert not app.exception
            assert app.metric[0].value


def test_case_search_empty_results():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    app.sidebar.radio[0].set_value("Case explorer").run()
    app.text_input[0].set_value("zzzzzzunmatchabletoken").run()
    assert not app.exception
    assert any("No matching reports" in item.value for item in app.info)


def test_hybrid_search_and_pagination():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    app.sidebar.radio[0].set_value("Case explorer").run()
    app.number_input[0].set_value(2).run()
    assert not app.exception
    app.text_input[0].set_value("stolen mobile phone").run()
    next(radio for radio in app.radio if radio.label == "Search method").set_value("Hybrid").run()
    assert not app.exception
    assert app.number_input[0].value == 1
    assert len(app.dataframe) > 0


def test_informal_input_review_and_stale_result_handling():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    app.sidebar.radio[0].set_value("Narrative lab").run()
    app.checkbox[0].uncheck().run()
    app.text_area[0].set_value("sir my moblie stoln from pocket in bus ystrday").run()
    app.button[0].click().run()
    assert not app.exception
    assert app.metric[0].value == "Theft"
    app.text_area[0].set_value("please help").run()
    assert any("text has changed" in item.value for item in app.info)
    app.button[0].click().run()
    assert not app.exception
    assert any("Needs review" in item.value for item in app.warning)
    app.text_area[0].set_value("12345").run()
    app.button[0].click().run()
    assert not app.exception
    assert any("plain-text description" in item.value for item in app.warning)
