"""
S3/MinIO configuration for Turkish Legal AI.

This module provides S3-compatible object storage configuration:
- AWS S3 support
- MinIO support (local development)
- Multipart upload handling
- Pre-signed URLs
- Bucket management
- File encryption at rest
- Lifecycle policies

Storage Structure:
    documents/          - Legal documents (PDF, DOCX)
    documents/raw/      - Original uploaded files
    documents/processed/ - Processed/parsed files
    embeddings/         - Vector embeddings
    exports/            - Generated reports/exports
    temp/               - Temporary files (auto-delete)
    backups/            - Database backups
"""
import asyncio
import mimetypes
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import urlparse

import boto3
from botocore.client import Config
from botocore.exceptions import BotoCoreError, ClientError

from backend.core.config.settings import settings
from backend.core.constants import (
    MAX_DOCUMENT_SIZE_MB,
    SUPPORTED_DOCUMENT_TYPES,
)


class S3Config:
    """
    S3/MinIO configuration and client management.
    
    Provides utilities for:
    - File upload/download
    - Pre-signed URLs
    - Bucket management
    - Multipart uploads
    - Server-side encryption
    """
    
    def __init__(self) -> None:
        """Initialize S3 configuration."""
        self._client: boto3.client | None = None
        self._resource: boto3.resource | None = None
        self._bucket_name = settings.S3_BUCKET_NAME
        
        # Multipart upload thresholds
        self._multipart_threshold = 100 * 1024 * 1024  # 100 MB
        self._multipart_chunksize = 10 * 1024 * 1024   # 10 MB
    
    def get_client(self) -> boto3.client:
        """
        Get S3 boto3 client.
        
        Returns:
            boto3.client: Configured S3 client
            
        Example:
            >>> s3 = s3_config.get_client()
            >>> s3.list_buckets()
        """
        if self._client is None:
            # Parse endpoint URL
            endpoint_url = str(settings.S3_ENDPOINT_URL)
            
            # Configure client
            config = Config(
                signature_version='s3v4',
                s3={
                    'addressing_style': 'path',  # path or virtual
                },
                retries={
                    'max_attempts': 3,
                    'mode': 'adaptive',
                },
            )
            
            self._client = boto3.client(
                's3',
                endpoint_url=endpoint_url,
                aws_access_key_id=settings.S3_ACCESS_KEY_ID,
                aws_secret_access_key=settings.S3_SECRET_ACCESS_KEY.get_secret_value(),
                region_name=settings.S3_REGION,
                config=config,
                use_ssl=settings.S3_USE_SSL,
            )
        
        return self._client
    
    def get_resource(self) -> boto3.resource:
        """
        Get S3 boto3 resource.
        
        Returns:
            boto3.resource: Configured S3 resource
        """
        if self._resource is None:
            endpoint_url = str(settings.S3_ENDPOINT_URL)
            
            self._resource = boto3.resource(
                's3',
                endpoint_url=endpoint_url,
                aws_access_key_id=settings.S3_ACCESS_KEY_ID,
                aws_secret_access_key=settings.S3_SECRET_ACCESS_KEY.get_secret_value(),
                region_name=settings.S3_REGION,
                use_ssl=settings.S3_USE_SSL,
            )
        
        return self._resource
    
    # =========================================================================
    # BUCKET MANAGEMENT
    # =========================================================================
    
    def ensure_bucket_exists(self) -> bool:
        """
        Ensure the configured bucket exists, create if not.
        
        Returns:
            bool: True if bucket exists or was created
            
        Raises:
            ClientError: If bucket creation fails
        """
        s3 = self.get_client()
        
        try:
            # Check if bucket exists
            s3.head_bucket(Bucket=self._bucket_name)
            return True
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code')
            
            if error_code == '404':
                # Bucket doesn't exist, create it
                try:
                    if settings.S3_REGION == 'us-east-1':
                        # us-east-1 doesn't need LocationConstraint
                        s3.create_bucket(Bucket=self._bucket_name)
                    else:
                        s3.create_bucket(
                            Bucket=self._bucket_name,
                            CreateBucketConfiguration={
                                'LocationConstraint': settings.S3_REGION
                            }
                        )
                    
                    # Enable versioning (recommended for production)
                    if settings.ENVIRONMENT == "production":
                        s3.put_bucket_versioning(
                            Bucket=self._bucket_name,
                            VersioningConfiguration={'Status': 'Enabled'}
                        )
                    
                    # Set lifecycle policy for temp files
                    self._set_lifecycle_policy()
                    
                    return True
                except ClientError as create_error:
                    raise create_error
            else:
                raise e
    
    def _set_lifecycle_policy(self) -> None:
        """
        Set lifecycle policy for automatic cleanup.
        
        Rules:
        - Delete temp/ files after 1 day
        - Delete exports/ after 7 days
        - Transition backups/ to cheaper storage after 30 days
        """
        s3 = self.get_client()
        
        lifecycle_config = {
            'Rules': [
                {
                    'Id': 'delete-temp-files',
                    'Status': 'Enabled',
                    'Prefix': 'temp/',
                    'Expiration': {'Days': 1},
                },
                {
                    'Id': 'delete-old-exports',
                    'Status': 'Enabled',
                    'Prefix': 'exports/',
                    'Expiration': {'Days': 7},
                },
                {
                    'Id': 'archive-old-backups',
                    'Status': 'Enabled',
                    'Prefix': 'backups/',
                    'Transitions': [
                        {
                            'Days': 30,
                            'StorageClass': 'GLACIER',
                        }
                    ],
                },
            ]
        }
        
        try:
            s3.put_bucket_lifecycle_configuration(
                Bucket=self._bucket_name,
                LifecycleConfiguration=lifecycle_config
            )
        except ClientError:
            # Lifecycle might not be supported (MinIO)
            pass
    
    def list_buckets(self) -> list[dict[str, Any]]:
        """
        List all S3 buckets.
        
        Returns:
            list: List of bucket information
        """
        s3 = self.get_client()
        
        try:
            response = s3.list_buckets()
            return response.get('Buckets', [])
        except ClientError:
            return []
    
    # =========================================================================
    # FILE UPLOAD
    # =========================================================================
    
    async def upload_file(
        self,
        file_data: bytes | BinaryIO,
        key: str,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
        encrypt: bool = True,
    ) -> dict[str, Any]:
        """
        Upload a file to S3.
        
        Args:
            file_data: File content (bytes or file-like object)
            key: S3 object key (path)
            content_type: MIME type (auto-detected if None)
            metadata: Custom metadata tags
            encrypt: Enable server-side encryption
            
        Returns:
            dict: Upload result with URL and metadata
            
        Example:
            >>> result = await s3_config.upload_file(
            ...     file_data=pdf_bytes,
            ...     key="documents/raw/contract_123.pdf",
            ...     metadata={"user_id": "123", "doc_type": "contract"}
            ... )
            >>> print(result["url"])
        """
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._upload_file_sync,
            file_data,
            key,
            content_type,
            metadata,
            encrypt,
        )
    
    def _upload_file_sync(
        self,
        file_data: bytes | BinaryIO,
        key: str,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
        encrypt: bool = True,
    ) -> dict[str, Any]:
        """Synchronous file upload implementation."""
        s3 = self.get_client()
        
        # Auto-detect content type
        if content_type is None:
            content_type = mimetypes.guess_type(key)[0] or 'application/octet-stream'
        
        # Prepare extra args
        extra_args: dict[str, Any] = {
            'ContentType': content_type,
        }
        
        # Add server-side encryption
        if encrypt:
            extra_args['ServerSideEncryption'] = 'AES256'
        
        # Add metadata
        if metadata:
            extra_args['Metadata'] = metadata
        
        # Convert bytes to BytesIO if needed
        if isinstance(file_data, bytes):
            file_obj = BytesIO(file_data)
        else:
            file_obj = file_data
        
        try:
            # Upload file
            s3.upload_fileobj(
                file_obj,
                self._bucket_name,
                key,
                ExtraArgs=extra_args,
            )
            
            # Get object info
            head = s3.head_object(Bucket=self._bucket_name, Key=key)
            
            return {
                'success': True,
                'key': key,
                'bucket': self._bucket_name,
                'size': head['ContentLength'],
                'content_type': head['ContentType'],
                'etag': head['ETag'].strip('"'),
                'last_modified': head['LastModified'],
                'url': self._get_object_url(key),
                'metadata': head.get('Metadata', {}),
            }
        except (ClientError, BotoCoreError) as e:
            return {
                'success': False,
                'error': str(e),
                'key': key,
            }
    
    async def upload_file_from_path(
        self,
        file_path: str | Path,
        key: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Upload a file from local filesystem.
        
        Args:
            file_path: Local file path
            key: S3 key (auto-generated from filename if None)
            **kwargs: Additional arguments for upload_file
            
        Returns:
            dict: Upload result
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {
                'success': False,
                'error': f'File not found: {file_path}',
            }
        
        # Auto-generate key from filename
        if key is None:
            key = f"documents/{file_path.name}"
        
        # Read file
        with open(file_path, 'rb') as f:
            file_data = f.read()
        
        return await self.upload_file(file_data, key, **kwargs)
    
    # =========================================================================
    # FILE DOWNLOAD
    # =========================================================================
    
    async def download_file(self, key: str) -> bytes | None:
        """
        Download a file from S3.
        
        Args:
            key: S3 object key
            
        Returns:
            bytes | None: File content or None if not found
            
        Example:
            >>> content = await s3_config.download_file("documents/contract.pdf")
            >>> if content:
            ...     with open("contract.pdf", "wb") as f:
            ...         f.write(content)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._download_file_sync,
            key,
        )
    
    def _download_file_sync(self, key: str) -> bytes | None:
        """Synchronous file download implementation."""
        s3 = self.get_client()
        
        try:
            buffer = BytesIO()
            s3.download_fileobj(self._bucket_name, key, buffer)
            buffer.seek(0)
            return buffer.read()
        except ClientError as e:
            if e.response.get('Error', {}).get('Code') == '404':
                return None
            raise
    
    async def download_file_to_path(
        self,
        key: str,
        local_path: str | Path,
    ) -> bool:
        """
        Download file directly to local filesystem.
        
        Args:
            key: S3 object key
            local_path: Local file path to save
            
        Returns:
            bool: True if successful
        """
        content = await self.download_file(key)
        
        if content is None:
            return False
        
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(local_path, 'wb') as f:
            f.write(content)
        
        return True
    
    # =========================================================================
    # PRE-SIGNED URLs
    # =========================================================================
    
    def generate_presigned_url(
        self,
        key: str,
        expires_in: int = 3600,
        http_method: str = 'GET',
    ) -> str:
        """
        Generate a pre-signed URL for temporary access.
        
        Args:
            key: S3 object key
            expires_in: URL expiration time in seconds (default: 1 hour)
            http_method: HTTP method (GET, PUT, POST)
            
        Returns:
            str: Pre-signed URL
            
        Example:
            >>> url = s3_config.generate_presigned_url(
            ...     "documents/contract.pdf",
            ...     expires_in=3600
            ... )
            >>> # Share URL with user for download
        """
        s3 = self.get_client()
        
        # Map method to boto3 operation
        operation_map = {
            'GET': 'get_object',
            'PUT': 'put_object',
            'POST': 'put_object',
        }
        
        operation = operation_map.get(http_method.upper(), 'get_object')
        
        try:
            url = s3.generate_presigned_url(
                operation,
                Params={
                    'Bucket': self._bucket_name,
                    'Key': key,
                },
                ExpiresIn=expires_in,
            )
            return url
        except (ClientError, BotoCoreError) as e:
            raise RuntimeError(f"Failed to generate pre-signed URL: {e}")
    
    def generate_upload_presigned_url(
        self,
        key: str,
        expires_in: int = 3600,
        max_size_mb: int = MAX_DOCUMENT_SIZE_MB,
    ) -> dict[str, Any]:
        """
        Generate pre-signed URL for direct upload from client.
        
        Args:
            key: S3 object key
            expires_in: URL expiration time in seconds
            max_size_mb: Maximum file size in MB
            
        Returns:
            dict: URL and upload fields
            
        Example:
            >>> upload_data = s3_config.generate_upload_presigned_url(
            ...     "documents/upload_123.pdf",
            ...     max_size_mb=100
            ... )
            >>> # Return to frontend for direct upload
        """
        s3 = self.get_client()
        
        conditions = [
            {'bucket': self._bucket_name},
            {'key': key},
            ['content-length-range', 1, max_size_mb * 1024 * 1024],
        ]
        
        try:
            response = s3.generate_presigned_post(
                Bucket=self._bucket_name,
                Key=key,
                Conditions=conditions,
                ExpiresIn=expires_in,
            )
            
            return {
                'url': response['url'],
                'fields': response['fields'],
                'expires_in': expires_in,
            }
        except (ClientError, BotoCoreError) as e:
            raise RuntimeError(f"Failed to generate upload URL: {e}")
    
    # =========================================================================
    # FILE OPERATIONS
    # =========================================================================
    
    async def delete_file(self, key: str) -> bool:
        """
        Delete a file from S3.
        
        Args:
            key: S3 object key
            
        Returns:
            bool: True if deleted successfully
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._delete_file_sync,
            key,
        )
    
    def _delete_file_sync(self, key: str) -> bool:
        """Synchronous file deletion."""
        s3 = self.get_client()
        
        try:
            s3.delete_object(Bucket=self._bucket_name, Key=key)
            return True
        except ClientError:
            return False
    
    async def file_exists(self, key: str) -> bool:
        """
        Check if a file exists in S3.
        
        Args:
            key: S3 object key
            
        Returns:
            bool: True if file exists
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._file_exists_sync,
            key,
        )
    
    def _file_exists_sync(self, key: str) -> bool:
        """Synchronous file existence check."""
        s3 = self.get_client()
        
        try:
            s3.head_object(Bucket=self._bucket_name, Key=key)
            return True
        except ClientError:
            return False
    
    async def get_file_metadata(self, key: str) -> dict[str, Any] | None:
        """
        Get file metadata without downloading.
        
        Args:
            key: S3 object key
            
        Returns:
            dict | None: File metadata or None if not found
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._get_file_metadata_sync,
            key,
        )
    
    def _get_file_metadata_sync(self, key: str) -> dict[str, Any] | None:
        """Synchronous metadata retrieval."""
        s3 = self.get_client()
        
        try:
            response = s3.head_object(Bucket=self._bucket_name, Key=key)
            
            return {
                'key': key,
                'size': response['ContentLength'],
                'content_type': response.get('ContentType'),
                'etag': response['ETag'].strip('"'),
                'last_modified': response['LastModified'],
                'metadata': response.get('Metadata', {}),
                'version_id': response.get('VersionId'),
            }
        except ClientError:
            return None
    
    async def list_files(
        self,
        prefix: str = '',
        max_keys: int = 1000,
    ) -> list[dict[str, Any]]:
        """
        List files in S3 with given prefix.
        
        Args:
            prefix: Key prefix to filter
            max_keys: Maximum number of keys to return
            
        Returns:
            list: List of file metadata
            
        Example:
            >>> files = await s3_config.list_files(prefix="documents/raw/")
            >>> for file in files:
            ...     print(file["key"], file["size"])
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._list_files_sync,
            prefix,
            max_keys,
        )
    
    def _list_files_sync(
        self,
        prefix: str = '',
        max_keys: int = 1000,
    ) -> list[dict[str, Any]]:
        """Synchronous file listing."""
        s3 = self.get_client()
        
        try:
            response = s3.list_objects_v2(
                Bucket=self._bucket_name,
                Prefix=prefix,
                MaxKeys=max_keys,
            )
            
            files = []
            for obj in response.get('Contents', []):
                files.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'],
                    'etag': obj['ETag'].strip('"'),
                })
            
            return files
        except ClientError:
            return []
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def _get_object_url(self, key: str) -> str:
        """
        Get the public URL for an object.
        
        Args:
            key: S3 object key
            
        Returns:
            str: Object URL
        """
        endpoint = str(settings.S3_ENDPOINT_URL)
        
        # Handle different URL formats
        if 'amazonaws.com' in endpoint:
            # AWS S3 format
            return f"https://{self._bucket_name}.s3.{settings.S3_REGION}.amazonaws.com/{key}"
        else:
            # MinIO or custom S3 format
            parsed = urlparse(endpoint)
            return f"{parsed.scheme}://{parsed.netloc}/{self._bucket_name}/{key}"
    
    async def check_health(self) -> dict[str, Any]:
        """
        Check S3 connection health.
        
        Returns:
            dict: Health status
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._check_health_sync)
    
    def _check_health_sync(self) -> dict[str, Any]:
        """Synchronous health check."""
        try:
            s3 = self.get_client()
            
            # Try to head the bucket
            start = datetime.now()
            s3.head_bucket(Bucket=self._bucket_name)
            latency = (datetime.now() - start).total_seconds() * 1000
            
            return {
                'healthy': True,
                'latency_ms': round(latency, 2),
                'bucket': self._bucket_name,
                'endpoint': str(settings.S3_ENDPOINT_URL),
            }
        except (ClientError, BotoCoreError) as e:
            return {
                'healthy': False,
                'error': str(e),
                'bucket': self._bucket_name,
            }
    
    def get_storage_info(self) -> dict[str, Any]:
        """
        Get storage usage information.
        
        Returns:
            dict: Storage statistics
        """
        s3 = self.get_client()
        
        try:
            # Get bucket size by listing all objects
            paginator = s3.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self._bucket_name)
            
            total_size = 0
            total_objects = 0
            
            for page in pages:
                for obj in page.get('Contents', []):
                    total_size += obj['Size']
                    total_objects += 1
            
            return {
                'bucket': self._bucket_name,
                'total_objects': total_objects,
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'total_size_gb': round(total_size / (1024 * 1024 * 1024), 2),
            }
        except ClientError as e:
            return {
                'error': str(e),
            }


# =============================================================================
# GLOBAL S3 CONFIG INSTANCE
# =============================================================================

s3_config = S3Config()


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "S3Config",
    "s3_config",
]