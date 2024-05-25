# -*- coding: utf-8 -*-
"""Bot

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1OhQ15Dj3WwISrv6u_CecbmTe127uwwUd
"""

import streamlit as st
from transformers import pipeline, set_seed


@st.cache(allow_output_mutation=True)
def load_generate():
    text_generator = pipeline("text-generation", model="openai-gpt")
    set_seed(42)
    return text_generator


generator = load_generate()

st.title("Бот для дополнения текста")
text = st.text_area("Место для записи начала текста", height=100)

num_sequences = st.slider("Количество предложений:", min_value=1, max_value=10)

len_sequences = st.slider("Длина предложений:", min_value=10, max_value=100)

generate_button = st.button("Дополнить текст")

def generate_text(text, max_length, num_sequences):
    if len(text) > 0:
        results = generator(text, max_length=max_length, num_return_sequences=num_sequences)
        for result in results:
            st.write(result["generated_text"])
    else:
        st.write("Вы не ввели текст")

if generate_button:
    st.write("**Варианты продолжения текста :**")
    generate_text(text, len_sequences, num_sequences)
