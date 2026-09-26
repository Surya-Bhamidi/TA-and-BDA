"""Optional real-browser verification and PDF export (requires Playwright and Edge)."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main():
    screenshots = ROOT / "docs" / "screenshots"
    screenshots.mkdir(exist_ok=True)
    errors, visited = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1080}, device_scale_factor=1)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto("http://127.0.0.1:8501", wait_until="domcontentloaded")
        page.get_by_role("heading", name="From narratives to insight", exact=True).wait_for(timeout=90000)
        page.wait_for_timeout(2000)
        page.screenshot(path=str(screenshots / "overview.png"), full_page=True)
        visited.append("Overview")
        for name, title, shot in [("Case explorer", "Explore the case library", "case-explorer.png"), ("Model evaluation", "Measure what the models learn", "evaluation.png"), ("Topics", "Discover recurring language", "topics.png")]:
            page.get_by_text(name, exact=True).click()
            page.get_by_role("heading", name=title, exact=True).wait_for(timeout=90000)
            page.wait_for_timeout(1500)
            assert page.locator('[data-testid="stException"]').count() == 0
            page.screenshot(path=str(screenshots / shot), full_page=True)
            visited.append(name)
        # Restore the landing page for the user's next visit.
        page.get_by_text("Overview", exact=True).click()
        page.get_by_role("heading", name="From narratives to insight", exact=True).wait_for()
        assert not errors, errors
        report = ROOT / "docs" / "PROJECT_REPORT.html"
        if report.exists():
            page.goto(report.as_uri())
            page.pdf(path=str(ROOT / "docs" / "PROJECT_REPORT.pdf"), format="A4", print_background=True, margin={"top": "16mm", "bottom": "16mm", "left": "16mm", "right": "16mm"})
        browser.close()
    result = {"status": "passed", "pages": visited, "javascript_errors": errors, "screenshots": str(screenshots)}
    (ROOT / "artifacts" / "browser_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
