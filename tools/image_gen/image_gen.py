from dotenv import load_dotenv
from openai import OpenAI
import os
import time
import base64
from pydantic import BaseModel, Field
from langchain.agents import tool
from langchain.output_parsers import PydanticOutputParser
from tools.file_storage.file_storage import store_file, get_image_file_from_url







def generateDalle3Image(query: str):

    load_dotenv()

    client = OpenAI()

    response = client.images.generate(
    model="dall-e-3",
    prompt=query,
    size="1024x1024",
    quality="standard",
    response_format="b64_json",
    n=1,
    )

    # print("Response:", response.data)
    # image_url = response.data[0].url
    # return image_url
    generated_image_base64 = response.data[0].b64_json
    generated_image_bytes = base64.b64decode(generated_image_base64)

    # Save the image to a unique file name
    file_name = f"generated_image_{int(time.time())}.png"
    temp_file_path = os.path.join("/tmp", file_name)  # Save to a temporary directory
    with open(temp_file_path, "wb") as f:
        f.write(generated_image_bytes)

    return temp_file_path


class EditImageStringsArgs(BaseModel):
    query: str = Field(description="this is the query to edit the image")
    image: str = Field(description="this is the image url")
    


def edit_tool_parser ():
    return PydanticOutputParser(pydantic_object=EditImageStringsArgs)

# SAMPLE PAYLOAD FOR EDITING IMAGE


# @tool(args_schema=EditImageStringsArgs,description="Useful for when you need to edit images when a previous image has been generated and the user wants to modify it",parse_docstring=True) 
def editDalle3Image(input: EditImageStringsArgs):

    load_dotenv()

    # Parse the input into an instance of EditImageStringsArgs
    if isinstance(input, str):
        input = EditImageStringsArgs.model_validate_json(input)


    client = OpenAI()

    # load image from url and store as a temporary file
    image_url = input.image
    image_file = get_image_file_from_url(image_url)
    # image_file = open(image_url, "rb")
    # Open the image file in binary mode
    with open(image_file, "rb") as image:
        response = client.images.edit(
            model="gpt-image-1",
            image=image,  # Pass the file directly, not as a list
            prompt=input.query,
            size="1024x1024",
            n=1,
        )

    # Extract the URL of the edited image
    print("Response:", response.data)
    editedImage = response.data[0].b64_json
    editedImageBytes = base64.b64decode(editedImage)
    # Save the edited image to a unique file name
    file_name = f"edited_image_{int(time.time())}.png"
    temp_file_path = os.path.join("/tmp", file_name)  # Save to a temporary directory
    with open(temp_file_path, "wb") as f:
        f.write(editedImageBytes)

    # Store the edited image in Firebase Storage
    file_url = store_file(temp_file_path)
    return file_url


def parser ():
        return PydanticOutputParser(pydantic_object=EditImageStringsArgs).get_format_instructions()

