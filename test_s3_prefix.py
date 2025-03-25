import boto3
from botocore.exceptions import ClientError
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env.local')

# AWS Configuration
S3_BUCKET_NAME = os.getenv('S3_BUCKET')
S3_PREFIX = os.getenv('S3_PREFIX')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION')

def test_s3_prefix():
    try:
        # Initialize S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_DEFAULT_REGION
        )
        
        print(f"Checking prefix '{S3_PREFIX}' in bucket '{S3_BUCKET_NAME}'")
        
        # List objects with the prefix
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=f"{S3_PREFIX}/"):
            if 'Contents' in page:
                print("\nFound objects:")
                for obj in page['Contents']:
                    print(f"- {obj['Key']} (Size: {obj['Size']} bytes)")
            else:
                print("\nNo objects found with this prefix")
                
    except ClientError as e:
        print(f"Error accessing S3: {e}")
        if e.response['Error']['Code'] == 'NoSuchBucket':
            print(f"Bucket '{S3_BUCKET_NAME}' does not exist")
        elif e.response['Error']['Code'] == 'AccessDenied':
            print("Access denied. Please check your AWS credentials and permissions")
        else:
            print(f"Unexpected error: {e}")

if __name__ == "__main__":
    test_s3_prefix() 