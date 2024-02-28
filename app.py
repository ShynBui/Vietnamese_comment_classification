import gradio as gr

import re
import os
import py_vncorenlp
from pyvi import ViTokenizer, ViPosTagger
import requests

def preprocess_text(text):
    # Loại bỏ các ký tự đặc biệt và dấu câu
    text = re.sub(r'[^\w\s]', '', text)

    # Loại bỏ URL
    text = re.sub(r'http\S+', '', text)

    # Loại bỏ đường dẫn file
    text = re.sub(r'\/\w+', '', text)

    return text

def remove_escape_sequences(text):
    escape_sequences = ['\n', '\t', '\r', '\\']
    for sequence in escape_sequences:
        text = text.replace(sequence, '')
    return text

def remove_html_tags(text):
    clean_text = re.sub(r'<[^>]*>', '', text)
    return clean_text

def vi_word_segment(text):
    output = ViTokenizer.tokenize(text)
    return output

def predict_comment(text):
    API_URL = "https://q6apanvp7xkd4yaj.us-east-1.aws.endpoints.huggingface.cloud"
    headers = {
        "Accept": "application/json",
        "Authorization": "Bearer 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",
        "Content-Type": "application/json"
    }

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    output = query({
        "inputs": text,
        "parameters": {
            "top_k": 1
        }
    })
    print(output)

    return output[0]['label']

def process_text(text):
    text = text[:256]
    text = preprocess_text(text)
    text = remove_escape_sequences(text)
    text = remove_escape_sequences(text)
    text = vi_word_segment(text)

    result = predict_comment(text)
    return result


if __name__ == '__main__':
    iface = gr.Interface(fn=process_text, inputs="text", outputs="text")
    iface.launch()