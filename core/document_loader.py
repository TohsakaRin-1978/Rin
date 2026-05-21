from pathlib import Path


def load_txt_files(folder_path: str):
    """Load all txt files from a folder."""
    documents = []
    folder = Path(folder_path)
    if not folder.exists():
        return documents
    for file in sorted(folder.glob("*.txt")):
        documents.append({"filename": file.name, "content": file.read_text(encoding="utf-8")})
    return documents


def load_uploaded_txt(uploaded_file):
    """Load txt content from a Streamlit uploaded file."""
    content = uploaded_file.read()
    if isinstance(content, bytes):
        return content.decode("utf-8")
    return str(content)
