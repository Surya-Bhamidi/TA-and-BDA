"""Verify exact user regressions in the actual browser and save UI evidence."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]


def main():
    cases = json.loads((ROOT / "artifacts/entity_context_regressions.json").read_text(encoding="utf-8"))["cases"]
    screenshots = ROOT / "docs/screenshots"
    screenshots.mkdir(exist_ok=True)
    verified = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1050})
        page.goto("http://127.0.0.1:8502", wait_until="domcontentloaded")
        page.get_by_text("Narrative lab", exact=True).click(timeout=90000)
        page.get_by_text("Include BERT sequence classification", exact=True).click()
        for i in [0, 1, 4]:
            case = cases[i]
            page.get_by_label("Case narrative", exact=True).fill(case["text"])
            page.get_by_role("button", name="Analyze narrative", exact=True).click()
            expect(page.get_by_text("Annotated narrative", exact=True)).to_be_visible(timeout=90000)
            expect(page.locator("mark")).to_have_count(len(case["entities"]), timeout=90000)
            actual = page.locator("mark").evaluate_all("""elements => elements.map(el => {
                const copy = el.cloneNode(true); const label = copy.querySelector('small').textContent;
                copy.querySelector('small').remove(); return {text: copy.textContent, label};
            })""")
            assert actual == [{"text": e["text"], "label": e["label"]} for e in case["entities"]], actual
            expect(page.get_by_label("Case narrative", exact=True)).to_have_value(case["text"])
            source = page.locator("mark").first.locator("..").evaluate("""el => {
                const copy = el.cloneNode(true); copy.querySelectorAll('small').forEach(e => e.remove()); return copy.textContent;
            }""")
            assert source == case["text"], (source, case["text"])
            assert page.get_by_test_id("stException").count() == 0
            assert "Release 3.1.0" in page.inner_text("body")
            page.locator("mark").first.locator("..").screenshot(path=str(screenshots / f"user-example-{i + 1}.png"))
            verified.append({"case": i + 1, "exact_entities": True, "exact_source_text": True})
        # Literal Markdown, bullet punctuation, Unicode and blank lines survive
        # the actual browser path, not just Python HTML parsing.
        text = "Mayank stole Surya's phone.\n\n1. Date: 29-06-2026\n2. Note: ₹3,500/- & <details> **literal**"
        page.get_by_label("Case narrative", exact=True).fill(text)
        page.get_by_role("button", name="Analyze narrative", exact=True).click()
        expect(page.get_by_text("Annotated narrative", exact=True)).to_be_visible(timeout=90000)
        page.wait_for_function("""expected => {
            const mark = document.querySelector('mark'); if (!mark) return false;
            const copy = mark.parentElement.cloneNode(true);
            copy.querySelectorAll('small').forEach(e => e.remove()); return copy.textContent === expected;
        }""", arg=text, timeout=90000)
        source = page.locator("mark").first.locator("..").evaluate("""el => {
            const copy = el.cloneNode(true); copy.querySelectorAll('small').forEach(e => e.remove()); return copy.textContent;
        }""")
        assert source == text, (source, text)
        browser.close()
    receipt = {"status": "passed", "release": "3.1.0", "screenshots": str(screenshots), "cases": verified,
               "literal_characters_and_line_breaks": True}
    (ROOT / "artifacts/user_ui_check.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
