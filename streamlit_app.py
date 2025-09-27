#import streamlit as st
#x = st.slider("testing 1234")
#st.write(x, "squared is", x * x)




# app.py
#!pip install fastapi "uvicorn[standard]" openai

import sys
import subprocess
'''
def install_package(package_name):
    """
    Installs a specified Python package using pip.
    """
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"Successfully installed {package_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error installing {package_name}: {e}")

# Example usage:
install_package("fastapi")
#install_package("pandas==1.3.4") # Install a specific version
'''

import os
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAI

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use env var for security: export OPENAI_API_KEY="sk-..."
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.post("/api/chatpdf")
async def chat_with_pdf(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    model: str = Form("gpt-4.1-mini")  # good balance of cost/quality
):
    if file.content_type not in ("application/pdf",):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    try:
        # 1) Upload the PDF to OpenAI Files
        #    (the SDK streams the file; no need to read into memory)
        uploaded = client.files.create(
            file=(file.filename, await file.read(), "application/pdf")
        )

        # 2) Call Responses API with the uploaded file referenced as input_file
        #    We send both the user prompt and the PDF in one request.
        resp = client.responses.create(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_file", "file_id": uploaded.id},
                    ],
                }
            ],
        )

        # 3) Extract the text output
        #    (Responses API returns a structured array of outputs)
        text_chunks = []
        if hasattr(resp, "output") and isinstance(resp.output, list):
            for item in resp.output:
                if item.get("type") == "message":
                    for part in item["content"]:
                        if part.get("type") == "output_text":
                            text_chunks.append(part.get("text", ""))
        result_text = "\n".join(text_chunks).strip() or "(No text output)"

        return JSONResponse({"answer": result_text})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:chat_with_pdf", host="0.0.0.0", port=8000, reload=True)
