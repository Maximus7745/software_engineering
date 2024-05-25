from fastapi import FastAPI, HTTPException
from transformers import pipeline, set_seed
from pydantic import BaseModel


class Item(BaseModel):
    text: str


app = FastAPI()
generator = pipeline("text-generation", model="openai-gpt")
set_seed(42)


@app.get("/")
async def root():
    return {"message": "hello everyone, input your message."}


@app.get("/info/")
async def get_info():
    info = {
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
    return info


@app.post("/generate/")
def generate_text(item: Item):
    if not item.text.strip():  # Проверка на пустую строку
        raise HTTPException(status_code=422, detail="Input text cannot be empty")
    return generator(item.text, max_length=20, num_return_sequences=5, bos_token_id=generator.model.config.bos_token_id)
