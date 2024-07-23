import pytest
from fastapi.testclient import TestClient
from main import app  # assuming your FastAPI app is named `app` in a file `my_app.py`

client = TestClient(app)

def test_upload_pdf():
    with open("sample.pdf", "rb") as f:
        response = client.post("/upload-pdf/", files={"file": f})
    assert response.status_code == 200
    assert response.json() == {"message": "PDF processed and vectors created"}

def test_ask_question():
    response = client.post("/ask-question/", json={"question": "What is this PDF about?"})
    assert response.status_code == 200
    assert "answer" in response.json()
