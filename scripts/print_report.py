"""Print the already generated HTML report to PDF using Playwright and installed Edge."""
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = ROOT / "docs" / "PROJECT_REPORT.html"
    if not report.exists():
        raise SystemExit("Run scripts/export_results.py first")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1100, "height": 1200})
        page.goto(report.as_uri())
        assert page.locator("h2").count() >= 17
        assert "Executed project results" in page.inner_text("body")
        page.pdf(path=str(ROOT / "docs" / "PROJECT_REPORT.pdf"), format="A4", print_background=True,
                 display_header_footer=True, header_template="<div></div>",
                 footer_template='<div style="font-size:8px;width:100%;text-align:center;color:#667">Decoding Crime Narratives · Version 2 · <span class="pageNumber"></span> / <span class="totalPages"></span></div>',
                 margin={"top": "16mm", "bottom": "19mm", "left": "16mm", "right": "16mm"})
        page.screenshot(path=str(ROOT / "docs" / "screenshots" / "report-preview.png"))
        browser.close()
    print("Printed docs/PROJECT_REPORT.pdf")


if __name__ == "__main__":
    main()
