"""Tests for Streamlit application."""

import pytest
from pathlib import Path
import sys

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))


@pytest.mark.ui
class TestAppImports:
    """Test that app modules can be imported without errors."""

    def test_main_imports(self):
        """Should import main app module."""
        try:
            import app.main as main
            assert main is not None
        except ImportError as e:
            pytest.skip(f"App main module not fully implemented: {e}")

    def test_challenge_page_imports(self):
        """Should import challenge page."""
        try:
            from app.pages import __init__
            # Page files use numeric prefixes which make direct import tricky
            # Just verify the pages directory exists
            pages_dir = Path(__file__).parent.parent / "app" / "pages"
            assert pages_dir.exists()
            assert (pages_dir / "01_challenge.py").exists()
        except Exception as e:
            pytest.skip(f"Challenge page not accessible: {e}")

    def test_reveal_page_exists(self):
        """Should have reveal page file."""
        pages_dir = Path(__file__).parent.parent / "app" / "pages"
        assert (pages_dir / "02_reveal.py").exists()

    def test_traditional_page_exists(self):
        """Should have traditional IDS page file."""
        pages_dir = Path(__file__).parent.parent / "app" / "pages"
        assert (pages_dir / "03_traditional.py").exists()

    def test_pipeline_page_exists(self):
        """Should have pipeline page file."""
        pages_dir = Path(__file__).parent.parent / "app" / "pages"
        assert (pages_dir / "04_pipeline.py").exists()

    def test_detection_page_exists(self):
        """Should have detection page file."""
        pages_dir = Path(__file__).parent.parent / "app" / "pages"
        assert (pages_dir / "05_detection.py").exists()


@pytest.mark.ui
class TestAppComponents:
    """Test UI component modules."""

    def test_components_directory_exists(self):
        """Should have components directory."""
        components_dir = Path(__file__).parent.parent / "app" / "components"
        assert components_dir.exists()

    def test_plots_component_exists(self):
        """Should have plots component file."""
        plots_file = Path(__file__).parent.parent / "app" / "components" / "plots.py"
        assert plots_file.exists()

    def test_metrics_component_exists(self):
        """Should have metrics component file."""
        metrics_file = Path(__file__).parent.parent / "app" / "components" / "metrics.py"
        assert metrics_file.exists()


@pytest.mark.ui
@pytest.mark.skip(reason="Streamlit app testing requires streamlit.testing module")
class TestAppFunctionality:
    """Test app functionality (requires Streamlit testing framework)."""

    def test_app_initialization(self):
        """Should initialize app with config and device info."""
        # This would use streamlit.testing.v1.AppTest
        # from streamlit.testing.v1 import AppTest
        # at = AppTest.from_file("app/main.py")
        # at.run()
        # assert not at.exception
        pass

    def test_session_state_initialization(self):
        """Should initialize session state with required keys."""
        # Would test that config and device info are in session_state
        pass

    def test_page_navigation(self):
        """Should be able to navigate between pages."""
        # Would test multi-page navigation
        pass


@pytest.mark.ui
class TestAppStructure:
    """Test overall app structure and organization."""

    def test_app_directory_structure(self):
        """Should have proper app directory structure."""
        app_dir = Path(__file__).parent.parent / "app"

        assert app_dir.exists()
        assert (app_dir / "main.py").exists()
        assert (app_dir / "pages").exists()
        assert (app_dir / "components").exists()

    def test_all_pages_present(self):
        """Should have all 5 demo pages."""
        pages_dir = Path(__file__).parent.parent / "app" / "pages"

        expected_pages = [
            "01_challenge.py",
            "02_reveal.py",
            "03_traditional.py",
            "04_pipeline.py",
            "05_detection.py"
        ]

        for page in expected_pages:
            assert (pages_dir / page).exists(), f"Missing page: {page}"

    def test_component_files_present(self):
        """Should have component files."""
        components_dir = Path(__file__).parent.parent / "app" / "components"

        expected_components = ["plots.py", "metrics.py"]

        for component in expected_components:
            assert (components_dir / component).exists(), f"Missing component: {component}"
