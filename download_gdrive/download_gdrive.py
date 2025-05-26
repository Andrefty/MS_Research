import os
import io
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError

# If modifying these SCOPES, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
FOLDER_ID = '1cznxGme5o6A_9tT8T47JUh3MPEpRYiKK' # The ID of the folder you provided
DOWNLOAD_PATH = './PrimeVul-v0.1' # Download files to the current directory

def authenticate():
    """Shows basic usage of the Drive v3 API.
    Prints the names and ids of the first 10 files the user has access to.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Make sure you have 'credentials.json' in the same directory as the script
            # This file is obtained from Google Cloud Console for your project.
            if not os.path.exists('credentials.json'):
                print("Error: 'credentials.json' not found. Please download it from Google Cloud Console and place it in the script's directory.")
                print("Instructions: ")
                print("1. Go to https://console.cloud.google.com/apis/credentials")
                print("2. Create or select a project.")
                print("3. Click on '+ CREATE CREDENTIALS' and choose 'OAuth client ID'.")
                print("4. Select 'Desktop app' as the Application type.")
                print("5. Name it (e.g., 'Drive API Downloader').")
                print("6. Click 'Create'.")
                print("7. Click 'DOWNLOAD JSON' for the client ID you just created. Rename the downloaded file to 'credentials.json' and place it in the same directory as this script.")
                print("8. You also need to enable the Google Drive API for your project: https://console.cloud.google.com/apis/library/drive.googleapis.com")
                return None
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds

def download_files_from_folder(folder_id, download_path):
    creds = authenticate()
    if not creds:
        print("Authentication failed. Exiting.")
        return

    try:
        service = build('drive', 'v3', credentials=creds)

        # Query to list files directly under the folder_id, excluding sub-folders
        query = f"'{folder_id}' in parents and mimeType != 'application/vnd.google-apps.folder'"
        
        page_token = None
        while True:
            results = service.files().list(
                q=query,
                spaces='drive',
                fields='nextPageToken, files(id, name)',
                pageToken=page_token
            ).execute()
            
            items = results.get('files', [])

            if not items:
                print('No files found at the first level of the folder.')
                break
            else:
                print('Files found:')
                for item in items:
                    file_id = item['id']
                    file_name = item['name']
                    print(f"- {file_name} (ID: {file_id})")
                    
                    request = service.files().get_media(fileId=file_id)
                    file_path = os.path.join(download_path, file_name)
                    
                    # Create download directory if it doesn't exist
                    os.makedirs(download_path, exist_ok=True)
                    
                    print(f"Downloading {file_name} to {file_path}...")
                    fh = io.FileIO(file_path, 'wb')
                    downloader = MediaIoBaseDownload(fh, request)
                    done = False
                    while done is False:
                        status, done = downloader.next_chunk()
                        print(f"Download {int(status.progress() * 100)}%.")
                    print(f"Finished downloading {file_name}.")

            page_token = results.get('nextPageToken', None)
            if page_token is None:
                break
        print("\\nAll files from the first level have been processed.")

    except HttpError as error:
        print(f'An API error occurred: {error}')
    except Exception as e:
        print(f'An unexpected error occurred: {e}')

if __name__ == '__main__':
    print(f"Attempting to download files from Google Drive Folder ID: {FOLDER_ID}")
    print(f"Files will be saved to: {os.path.abspath(DOWNLOAD_PATH)}\\n")
    download_files_from_folder(FOLDER_ID, DOWNLOAD_PATH)
