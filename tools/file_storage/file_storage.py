import firebase_admin
import os
import json
from google.oauth2 import service_account
import requests

from firebase_admin import credentials, storage



def store_file(image_file):
    """
    Store a file in Firebase Storage.

    Args:
        image_file (str): The path to the image file.

    Returns:
        str: The URL of the stored file.
    """
    # Strip any trailing whitespace or newline characters
    image_file = image_file.strip()

    # Initialize Firebase Admin SDK
    if not firebase_admin._apps:  # Check if Firebase app is already initialized
        if os.environ.get("ENV") == "production":
            # Use credentials from the environment variable
            credential_info = json.loads(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON"))
            cred = credentials.Certificate(credential_info)
        else:
            # Use local credentials in development
            cred = credentials.Certificate("/Users/enochaikpokpodion/Downloads/game-startup-ai-agent-4e9b990c6152.json")
        
        firebase_admin.initialize_app(cred, {
            'storageBucket': 'game-startup-ai-agent.firebasestorage.app'
        })

    # Get a reference to the storage bucket
    bucket = storage.bucket()

    # Format the file name
    formatted_file_name = format_file_name(os.path.basename(image_file))

    # Create a blob for the file
    blob = bucket.blob(formatted_file_name)

    # Upload the file to Firebase Storage
    blob.upload_from_filename(image_file)

    # Make the file publicly accessible
    blob.make_public()

    return blob.public_url
    
    


def format_file_name(file_name):
    """
    Format the file name to be used in Firebase Storage.

    Args:
        file_name (str): The original file name.

    Returns:
        str: The formatted file name.
    """
    # Remove spaces and special characters from the file name
    formatted_file_name = file_name.replace(" ", "_").replace("/", "_")
    return formatted_file_name


def get_image_file_from_url(url):

    print("URL:", url)
    """
    Get the image file from a URL.

    Args:
        url (str): The URL of the image file.

    Returns:
        str: The path to the downloaded image file.
    """
    # Download the image file from the URL
    response = requests.get(url)
    if response.status_code == 200:
        # Extract the file name and extension from the URL
        file_name = os.path.basename(url.split("?")[0])  # Remove query parameters
        if not file_name.endswith(('.png', '.jpg', '.jpeg', '.gif')):
            file_name += ".png"  # Default to .png if no valid extension is found

        # Save the image file to a temporary location
        temp_file_path = os.path.join("/tmp", file_name)
        with open(temp_file_path, "wb") as f:
            f.write(response.content)
        return temp_file_path
    else:
        return url
        # raise Exception(f"Failed to download image from {url}. HTTP status code: {response.status_code}")
    



# """
#     Store a file in Firebase Storage and Firestore.

#     Args:
#         file_path (str): The path to the file to be stored.
#         file_name (str): The name of the file to be stored in Firebase Storage.
#     """
#     if os.environ.get("ENV") == "production":
#         # Use credentials from the environment variable
#         credential_info = json.loads(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON"))
#         credential = service_account.Credentials.from_service_account_info(credential_info)
#     else:
#         # Exclude credentials in deployment
#         credential = None

#     # Initialize Firebase Admin SDK
#     cred = credentials.Certificate("/Users/enochaikpokpodion/Downloads/game-startup-ai-agent-4e9b990c6152.json")
#     firebase_admin.initialize_app(cred, {
#         'storageBucket': 'game-startup-ai-agent.firebasestorage.app'
#     })

#     # Get a reference to the storage bucket
#     bucket = storage.bucket()

#     # Upload the file to Firebase Storage
#     blob = bucket.blob(format_file_name(file_name))
#     blob.upload_from_filename(format_file_name(file_name))
#     # Make the blob publicly accessible
#     blob.make_public()

#     # Get the download URL
#     download_url = blob.public_url

#     # return the download URL
#     return download_url