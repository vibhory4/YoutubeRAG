"""Playwright end-to-end tests for the Chainlit channel gallery UI (app.py).

Each test class uses a module-scoped chainlit server fixture so the browser
process is shared across tests in that class and startup cost is paid once.
"""

import json
import os
import subprocess
import tempfile
import time

import pytest
import requests
from playwright.sync_api import Page, expect

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_CHAINLIT = os.path.join(REPO_DIR, ".venv", "bin", "chainlit")
PERSONAS_DIR = os.path.join(REPO_DIR, "data", "personas")
CHROMA_DIR = os.path.join(REPO_DIR, "data", "chroma_db")

# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------
SOIC_CHANNEL = "https://www.youtube.com/@SOICfinance"

_STATE_WITH_CHANNELS = {
    "channels": {
        SOIC_CHANNEL: {
            "added_at": "2026-03-08T00:00:00",
            "last_checked": "2026-03-08T00:00:00",
            "indexed_video_ids": ["dQw4w9WgXcQ"],
            "total_videos_found": 5,
            "total_videos_indexed": 5,
            "total_videos_failed": 0,
        }
    },
    "last_updated": "2026-03-08T00:00:00",
}

_STATE_EMPTY = {"channels": {}, "last_updated": "2026-03-08T00:00:00"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _wait_for_server(url: str, timeout: int = 20) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"Chainlit server at {url} did not become ready within {timeout}s")


def _start_chainlit(state_data: dict, port: int) -> tuple[subprocess.Popen, str, str]:
    """Write a temp state file and start chainlit. Returns (proc, url, state_path)."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(state_data, tmp)
    tmp.close()
    state_path = tmp.name

    env = {
        **os.environ,
        "STATE_FILE": state_path,
        "PERSONAS_DIR": PERSONAS_DIR,
        "CHROMA_PERSIST_DIR": CHROMA_DIR,
    }
    proc = subprocess.Popen(
        [VENV_CHAINLIT, "run", "app.py", "--port", str(port), "--headless"],
        cwd=REPO_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    url = f"http://localhost:{port}"
    _wait_for_server(url)
    return proc, url, state_path


# ---------------------------------------------------------------------------
# Server fixtures (module-scoped so all tests in the class share one server)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def server_with_channels():
    proc, url, state_path = _start_chainlit(_STATE_WITH_CHANNELS, port=8301)
    yield url
    proc.terminate()
    proc.wait(timeout=10)
    os.unlink(state_path)


@pytest.fixture(scope="module")
def server_empty():
    proc, url, state_path = _start_chainlit(_STATE_EMPTY, port=8302)
    yield url
    proc.terminate()
    proc.wait(timeout=10)
    os.unlink(state_path)


# ---------------------------------------------------------------------------
# Tests: empty state (no channels indexed)
# ---------------------------------------------------------------------------
class TestEmptyState:
    def test_no_channels_message_shown(self, page: Page, server_empty: str):
        page.goto(server_empty)
        expect(page.get_by_text("No channels indexed yet", exact=False)).to_be_visible(
            timeout=15_000
        )

    def test_index_command_shown(self, page: Page, server_empty: str):
        page.goto(server_empty)
        expect(page.get_by_text("python main.py index", exact=False)).to_be_visible(
            timeout=15_000
        )

    def test_typing_without_channels_returns_guidance(self, page: Page, server_empty: str):
        page.goto(server_empty)
        page.wait_for_selector("text=No channels indexed yet", timeout=15_000)
        textarea = page.locator("textarea").first
        textarea.fill("hello")
        textarea.press("Enter")
        expect(page.get_by_text("Please select a channel", exact=False)).to_be_visible(
            timeout=10_000
        )


# ---------------------------------------------------------------------------
# Tests: gallery with one channel
# ---------------------------------------------------------------------------
class TestChannelGallery:
    def test_welcome_heading_visible(self, page: Page, server_with_channels: str):
        page.goto(server_with_channels)
        expect(page.get_by_text("Welcome to YouTube RAG", exact=False)).to_be_visible(
            timeout=15_000
        )

    def test_channel_name_on_card(self, page: Page, server_with_channels: str):
        page.goto(server_with_channels)
        expect(page.get_by_text("SOIC Finance", exact=False)).to_be_visible(timeout=15_000)

    def test_topics_on_card(self, page: Page, server_with_channels: str):
        page.goto(server_with_channels)
        expect(page.get_by_text("Investing", exact=False)).to_be_visible(timeout=15_000)

    def test_tone_on_card(self, page: Page, server_with_channels: str):
        page.goto(server_with_channels)
        expect(page.get_by_text("analytical", exact=False)).to_be_visible(timeout=15_000)

    def test_chat_button_present(self, page: Page, server_with_channels: str):
        page.goto(server_with_channels)
        page.wait_for_selector("text=SOIC Finance", timeout=15_000)
        btn = page.get_by_role("button", name="Chat →")
        expect(btn).to_be_visible()


# ---------------------------------------------------------------------------
# Tests: selecting a channel
# ---------------------------------------------------------------------------
class TestChannelSelection:
    def test_click_chat_shows_agent_confirmation(self, page: Page, server_with_channels: str):
        page.goto(server_with_channels)
        page.wait_for_selector("text=Chat →", timeout=15_000)
        page.get_by_role("button", name="Chat →").click()
        expect(page.get_by_text("Now chatting with SOIC Finance", exact=False)).to_be_visible(
            timeout=30_000
        )

    def test_persona_summary_shown_after_selection(self, page: Page, server_with_channels: str):
        page.goto(server_with_channels)
        page.wait_for_selector("text=Chat →", timeout=15_000)
        page.get_by_role("button", name="Chat →").click()
        # persona_summary contains "stock market" per the persona file
        expect(page.get_by_text("stock market", exact=False)).to_be_visible(timeout=30_000)

    def test_starter_question_buttons_appear(self, page: Page, server_with_channels: str):
        page.goto(server_with_channels)
        page.wait_for_selector("text=Chat →", timeout=15_000)
        page.get_by_role("button", name="Chat →").click()
        page.wait_for_selector("text=Now chatting with SOIC Finance", timeout=30_000)
        expect(page.get_by_role("button", name="Tell me about Investing")).to_be_visible(
            timeout=10_000
        )

    def test_input_editable_after_selection(self, page: Page, server_with_channels: str):
        page.goto(server_with_channels)
        page.wait_for_selector("text=Chat →", timeout=15_000)
        page.get_by_role("button", name="Chat →").click()
        page.wait_for_selector("text=Now chatting with SOIC Finance", timeout=30_000)
        textarea = page.locator("textarea").first
        expect(textarea).to_be_editable()
        textarea.fill("What stocks do you follow?")
        expect(textarea).to_have_value("What stocks do you follow?")

    def test_typing_without_selecting_prompts_card_click(
        self, page: Page, server_with_channels: str
    ):
        page.goto(server_with_channels)
        page.wait_for_selector("text=SOIC Finance", timeout=15_000)
        textarea = page.locator("textarea").first
        textarea.fill("hello")
        textarea.press("Enter")
        expect(page.get_by_text("Please select a channel", exact=False)).to_be_visible(
            timeout=10_000
        )
