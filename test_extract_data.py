import os
import importlib
import json
from types import SimpleNamespace
from fastapi.testclient import TestClient

# Ensure FastAPI app reloads with correct EXTRACT_METHOD
def reload_extract_data(method):
    os.environ['EXTRACT_METHOD'] = method
    import extract_data
    importlib.reload(extract_data)
    return extract_data


def test_ner_cpu_extraction():
    ed = reload_extract_data('ner')
    client = TestClient(ed.app)
    response = client.post('/extract', json={'text': 'Intel i7-9700K'})
    assert response.status_code == 200
    assert response.json() == {'data': [['Intel i7-9700K', 'CPU']]}  # CPU extraction


def test_ner_gpu_extraction():
    ed = reload_extract_data('ner')
    client = TestClient(ed.app)
    response = client.post('/extract', json={'text': 'GTX 3060 Ti'})
    assert response.status_code == 200
    assert response.json() == {'data': [['GTX 3060 Ti', 'GraphicsCard']]}  # GPU extraction


def test_llm_component_extraction(monkeypatch):
    ed = reload_extract_data('llm')
    # Stub llm.invoke to return JSON array with the same text and label CPU
    def fake_invoke(messages):
        user_text = messages[1]['content']
        return SimpleNamespace(content=json.dumps([[user_text, 'CPU']]))
    monkeypatch.setattr(ed.llm, 'invoke', fake_invoke)
    client = TestClient(ed.app)
    text = 'Custom CPU Input'
    response = client.post('/extract', json={'text': text})
    assert response.status_code == 200
    assert response.json() == {'data': [[text, 'CPU']]}
