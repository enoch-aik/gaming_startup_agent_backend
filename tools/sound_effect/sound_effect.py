import os
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import play
from tools.file_storage.file_storage import store_file
from uuid import uuid4




def generate_sound_effect(query: str):
    """
    Generate a sound effect using ElevenLabs API.
    Args:
        query (str): The text prompt for the sound effect.
    Returns:
        str: The URL of the generated sound effect.
    """
    # Load environment variables from .env file
    load_dotenv()

    client = ElevenLabs(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
    )
    audio = client.text_to_sound_effects.convert(text=query)

    # Create a unique UUID as the sound effect name
    file_name = f"sound_effects/{uuid4()}.mp3"

    # Ensure the directory exists
    temp_dir = os.path.join("/tmp", "sound_effects")
    os.makedirs(temp_dir, exist_ok=True)

    # Save the audio to a temporary file
    temp_file_path = os.path.join(temp_dir, os.path.basename(file_name))
    with open(temp_file_path, "wb") as f:
        for chunk in audio:  # Iterate over the generator
            f.write(chunk)

    # Upload the file to Firebase Storage
    url = store_file(temp_file_path)
    return url




