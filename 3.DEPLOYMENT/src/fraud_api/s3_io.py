"""S3 access. Identity comes ONLY from the default credential chain (IRSA in the pod, ~/.aws locally).
Never pass keys here: explicit keys override and disable the pod's scoped role."""
from __future__ import annotations

import hashlib

import boto3
from botocore.exceptions import ClientError

MISSING_CODES = {"404", "NoSuchKey", "NotFound"}


def client(region: str):
    return boto3.client("s3", region_name=region)


def get_bytes(s3, bucket: str, key: str) -> bytes:
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read()


def exists(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in MISSING_CODES:
            return False
        raise


def put_bytes(s3, bucket: str, key: str, body: bytes, content_type: str = "application/json") -> None:
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType=content_type)


def sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()
