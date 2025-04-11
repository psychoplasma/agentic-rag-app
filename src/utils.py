import vertexai

from google.oauth2 import service_account


def authenticate_vertex_ai(project: str, location: str, credentials_path: str, bucket_uri: str):
    if not project:
        raise ValueError("project id is empty")

    if not location:
        raise ValueError("location is empty")

    if not credentials_path:
        raise ValueError("credentials path is empty")
    
    if not bucket_uri:
        raise ValueError("bucket uri is empty")

    credentials = service_account.Credentials.from_service_account_file(
        credentials_path,
        scopes=['https://www.googleapis.com/auth/cloud-platform']
    )

    vertexai.init(
        project=project,
        location=location,
        credentials=credentials,
        staging_bucket=bucket_uri,
    )
