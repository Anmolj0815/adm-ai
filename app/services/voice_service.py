import os
import base64
from dotenv import load_dotenv
from elevenlabs import generate, set_api_key

load_dotenv()

class VoiceService:
    def __init__(self):
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        if self.elevenlabs_api_key:
            set_api_key(self.elevenlabs_api_key)
        else:
            print("ELEVENLABS_API_KEY not found. Text-to-speech will be mocked.")

    async def speech_to_text(self, audio_data_base64: str) -> str:
        print("Mocking speech to text conversion...")
        return "tell me about the admission criteria for computer science engineering"

    async def text_to_speech(self, text: str) -> str:
        print(f"Converting text to speech: {text}")
        if self.elevenlabs_api_key:
            try:
                audio = generate(
                    text=text,
                    voice="Rachel",
                    model="eleven_multilingual_v2",
                    api_key=self.elevenlabs_api_key
                )
                return "https://example.com/generated_audio.mp3"
            except Exception as e:
                print(f"Error with ElevenLabs TTS: {e}")
                return "Error generating voice output."
        else:
            print("ElevenLabs API key not set. Voice output mocked.")
            return "https://example.com/mock_audio.mp3"
