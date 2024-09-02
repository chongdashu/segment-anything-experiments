import urllib.request
import zipfile
import os

def download_example_video():
    # Set the current directory to the script's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Create a 'videos' directory if it doesn't exist
    os.makedirs('videos', exist_ok=True)

    # Download the video
    url = "https://dl.fbaipublicfiles.com/segment_anything_2/assets/bedroom.zip"
    zip_path = os.path.join('videos', 'bedroom.zip')
    print("Downloading video...")
    urllib.request.urlretrieve(url, zip_path)

    # Extract the zip file
    print("Extracting video...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('videos')

    # Remove the zip file
    os.remove(zip_path)

    print("Video downloaded and extracted to 'videos/bedroom' directory.")

if __name__ == "__main__":
    download_example_video()
