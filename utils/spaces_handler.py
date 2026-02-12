"""
Digital Ocean Spaces handler for uploading and downloading files.
Uses boto3 (S3-compatible API).
"""
import boto3
from botocore.exceptions import ClientError
import logging
from pathlib import Path
from typing import Optional
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpacesHandler:
    """Handler for Digital Ocean Spaces operations."""
    
    def __init__(self):
        """Initialize the Spaces client."""
        self.client = boto3.client(
            "s3",
            region_name=config.DO_SPACES_CONFIG["region_name"],
            endpoint_url=config.DO_SPACES_CONFIG["endpoint_url"],
            aws_access_key_id=config.DO_SPACES_CONFIG["aws_access_key_id"],
            aws_secret_access_key=config.DO_SPACES_CONFIG["aws_secret_access_key"],
        )
        self.bucket = config.DO_SPACES_BUCKET
        
    def upload_file(
        self, 
        local_path: str, 
        remote_path: str, 
        make_public: bool = False
    ) -> bool:
        """
        Upload a file to Digital Ocean Spaces.
        
        Args:
            local_path: Local file path
            remote_path: Remote path in the bucket
            make_public: Whether to make the file publicly accessible
            
        Returns:
            True if successful, False otherwise
        """
        try:
            extra_args = {}
            if make_public:
                extra_args["ACL"] = "public-read"
                
            self.client.upload_file(
                local_path, 
                self.bucket, 
                remote_path,
                ExtraArgs=extra_args
            )
            logger.info(f"Successfully uploaded {local_path} to {remote_path}")
            return True
            
        except ClientError as e:
            logger.error(f"Error uploading file: {e}")
            return False
            
    def download_file(
        self, 
        remote_path: str, 
        local_path: str
    ) -> bool:
        """
        Download a file from Digital Ocean Spaces.
        
        Args:
            remote_path: Remote path in the bucket
            local_path: Local file path to save to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            
            self.client.download_file(
                self.bucket, 
                remote_path, 
                local_path
            )
            logger.info(f"Successfully downloaded {remote_path} to {local_path}")
            return True
            
        except ClientError as e:
            logger.error(f"Error downloading file: {e}")
            return False
            
    def list_files(self, prefix: str = "") -> list:
        """
        List files in the bucket with a given prefix.
        
        Args:
            prefix: Prefix to filter files
            
        Returns:
            List of file keys
        """
        try:
            response = self.client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix
            )
            
            if "Contents" in response:
                return [obj["Key"] for obj in response["Contents"]]
            return []
            
        except ClientError as e:
            logger.error(f"Error listing files: {e}")
            return []
            
    def file_exists(self, remote_path: str) -> bool:
        """
        Check if a file exists in the bucket.
        
        Args:
            remote_path: Remote path in the bucket
            
        Returns:
            True if file exists, False otherwise
        """
        try:
            self.client.head_object(Bucket=self.bucket, Key=remote_path)
            return True
        except ClientError:
            return False
            
    def get_public_url(self, remote_path: str) -> str:
        """
        Get the public URL for a file in Spaces.
        
        Args:
            remote_path: Remote path in the bucket
            
        Returns:
            Public URL of the file
        """
        endpoint = config.DO_SPACES_CONFIG["endpoint_url"].replace("https://", "")
        return f"https://{self.bucket}.{endpoint}/{remote_path}"


# Convenience functions
def upload_data_file(local_path: str, filename: str) -> bool:
    """Upload a data file to Spaces."""
    handler = SpacesHandler()
    remote_path = f"{config.DO_SPACES_DATA_PREFIX}{filename}"
    return handler.upload_file(local_path, remote_path)


def download_data_file(filename: str, local_path: str) -> bool:
    """Download a data file from Spaces."""
    handler = SpacesHandler()
    remote_path = f"{config.DO_SPACES_DATA_PREFIX}{filename}"
    return handler.download_file(remote_path, local_path)


def upload_model(local_path: str, model_name: str) -> bool:
    """Upload a model file to Spaces."""
    handler = SpacesHandler()
    remote_path = f"{config.DO_SPACES_MODEL_PREFIX}{model_name}"
    return handler.upload_file(local_path, remote_path)


def download_model(model_name: str, local_path: str) -> bool:
    """Download a model file from Spaces."""
    handler = SpacesHandler()
    remote_path = f"{config.DO_SPACES_MODEL_PREFIX}{model_name}"
    return handler.download_file(remote_path, local_path)
