import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STYLES_PATH = PROJECT_ROOT / "frontend" / "src" / "styles.css"


def test_dark_theme_guard_is_last_background_authority():
    css = STYLES_PATH.read_text(encoding="utf-8")
    marker = "Dark console consistency guard"

    assert marker in css
    guarded_css = css.split(marker, 1)[1]
    light_backgrounds = re.findall(
        r"background(?:-color)?:\s*(?:#fff(?:fff)?|#f[0-9a-fA-F]{2,5}|#e[0-9a-fA-F]{2,5}|white)\b",
        guarded_css,
    )

    assert light_backgrounds == []


def test_high_risk_empty_states_are_in_dark_theme_guard():
    css = STYLES_PATH.read_text(encoding="utf-8")
    guarded_css = css.split("Dark console consistency guard", 1)[1]

    for selector in (
        ".app-shell .model-dir-list > .empty-state",
        ".app-shell .model-catalog > .empty-state",
        ".app-shell .panel > .empty-state",
        ".app-shell .context-list > .empty-state",
        ".app-shell .related-artifacts > .empty-state",
    ):
        assert selector in guarded_css


def test_high_risk_interactive_surfaces_are_in_dark_theme_guard():
    css = STYLES_PATH.read_text(encoding="utf-8")
    guarded_css = css.split("Dark console consistency guard", 1)[1]

    for selector in (
        ".app-shell .task-row:hover",
        ".app-shell .artifact-row:hover",
        ".app-shell .context-row:hover",
        ".app-shell .model-dir-row:hover",
        ".app-shell .candidate-card:hover",
        ".app-shell button.guide-card:hover",
    ):
        assert selector in guarded_css
