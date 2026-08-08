def test_api_file_exists():
    """Check that the API file exists."""
    from pathlib import Path

    api_files = [
        Path("api/main.py"),
        Path("api/app.py"),
        Path("app/main.py"),
        Path("app/app.py"),
    ]

    assert any(file.exists() for file in api_files)