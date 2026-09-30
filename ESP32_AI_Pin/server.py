import os
import base64
import urllib.parse
from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable not set.")

client = OpenAI()

app = FastAPI(title="Smart Desk Camera Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Transcript", "X-GPT-Response"]
)

@app.post("/process")
async def process_interaction(
    image: UploadFile = File(...), 
    audio: UploadFile = File(...)
):
    try:
        audio_bytes = await audio.read()
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=(audio.filename, audio_bytes)
        )
        user_prompt = transcription.text
        print(f"User Asked: {user_prompt}")

        image_bytes = await image.read()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        mime_type = image.content_type or "image/jpeg"
        data_url = f"data:{mime_type};base64,{base64_image}"

        vision_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": f"You are a helpful smart desk assistant looking through a desk camera. Answer concisely. User prompt: {user_prompt}"
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url}
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        ai_text_response = vision_response.choices[0].message.content
        print(f"AI Answer: {ai_text_response}")

        tts_response = client.audio.speech.create(
            model="tts-1",
            voice="alloy", 
            input=ai_text_response
        )
        tts_audio_data = tts_response.read()
        headers = {
            "X-Transcript": urllib.parse.quote(user_prompt),
            "X-GPT-Response": urllib.parse.quote(ai_text_response)
        }

        return Response(
            content=tts_audio_data, 
            media_type="audio/mpeg", 
            headers=headers
        )

    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)