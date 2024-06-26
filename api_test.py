from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello everyone, input your message."}


def test_generate_text():
    input_text = {"text": "This is a test."}
    response = client.post("/generate/", json=input_text)
    assert response.status_code == 200
    output_list = response.json()
    assert isinstance(output_list, list)
    assert len(output_list) > 0
    generated_text = output_list[0]["generated_text"]
    assert generated_text is not None


def test_generate_text_length():
    input_text = {"text": "This is a test."}
    response = client.post("/generate/", json=input_text)
    assert response.status_code == 200
    output_list = response.json()
    assert isinstance(output_list, list)
    assert len(output_list) > 0
    generated_texts = [result["generated_text"][:20] for result in output_list]
    for generated_text in generated_texts:
        assert (
            len(generated_text) <= 20
        ), f"Generated text is longer than 20 characters: {generated_text}"


def test_info():
    response = client.get("/info/")
    assert response.status_code == 200
    assert response.json() == {
        "model": "openai-gpt",
        "description": "This is a text-generation model from OpenAI.",
        "methods": {
            "/": "GET - Root endpoint, returns a welcome message.",
            "/generate/": "POST - Generates text based on the input text. \
                Parameters: text (str), num_sequences (int) \
                    , max_length (int).",
            "/info/": "GET - Returns information about \
                the model and available API methods.",
        },
    }


def test_generate_text_error_handling():
    input_text = {"text": ""}
    response = client.post("/generate/", json=input_text)
    assert response.status_code == 422  # 422 Unprocessable Entity для пустого текста

    input_text = {"text": "This is a test."}
    response = client.post("/generate/", json=input_text)
    assert response.status_code == 200
    output_list = response.json()
    assert isinstance(output_list, list)
    assert len(output_list) > 0
    for result in output_list:
        assert "generated_text" in result
