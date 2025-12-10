import os
import logging

def upload_to_huggingface(local_path, repo_id, token=None, commit_message="Update model", path_in_repo=None):
    """
    Uploads a file to the Hugging Face Hub with error handling.
    
    Args:
        local_path (str): Path to the local file (e.g., 'checkpoints/model.pth').
        repo_id (str): Hugging Face repo ID (e.g., 'username/repo-name').
        token (str, optional): HF API token. Defaults to os.getenv('HF_TOKEN').
        commit_message (str, optional): Message for the commit.
        path_in_repo (str, optional): Filename in the repo. Defaults to local filename.
    """
    # 1. Resolve Token
    token = token or os.getenv("HF_TOKEN")
    
    if not token:
        print(f"[HF] ⚠ Skipping upload: HF_TOKEN environment variable not set.")
        print(f"[HF] File saved locally at: {local_path}")
        return

    # 2. Resolve Remote Path
    if path_in_repo is None:
        path_in_repo = os.path.basename(local_path)

    # 3. Attempt Upload
    try:
        from huggingface_hub import upload_file
        
        print(f"[HF] ⤒ Uploading {local_path} to {repo_id}...")
        
        upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            commit_message=commit_message,
            token=token
        )
        print(f"[HF] ✓ Successfully uploaded to: https://huggingface.co/{repo_id}")
        
    except ImportError:
        print("[HF] ✘ Error: 'huggingface_hub' library is not installed. Run pip install huggingface_hub.")
    except Exception as e:
        print(f"[HF] ✘ Upload failed: {e}")
        print(f"[HF] File remains saved locally at: {local_path}")