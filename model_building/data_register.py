from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os


repo_id = "deepacsr/tourism-package-prediction" #Hugging face Space id for the Prediction application
repo_type = "dataset"

# Initialize API client to log into Huggingface
#Reding the secret token for Hugging face from the Git Hub enviromnet
api = HfApi(token=os.getenv("HF_TOKEN_TOURIST"))

#Step 1: Check if the space exists, else create it.
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")


#Uploading the file in to the HF: https://huggingface.co/datasets/deepacsr/tourism-package-prediction/
api.upload_folder(
    folder_path="./data",
    repo_id=repo_id,
    repo_type=repo_type,
)
