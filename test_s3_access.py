import boto3
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env.local')

# Get AWS credentials from environment variables
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_region = os.getenv('AWS_DEFAULT_REGION')
bucket_name = os.getenv('S3_BUCKET')
prefix = os.getenv('S3_PREFIX')

print(f"Using bucket: {bucket_name}")
print(f"Using prefix: {prefix}")
print(f"Using region: {aws_region}")

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region
)

try:
    # Try to list objects in the bucket
    paginator = s3_client.get_paginator('list_objects_v2')
    total_objects = 0
    parquet_files = 0
    
    print("\nListing objects in bucket:")
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                total_objects += 1
                if obj['Key'].endswith('.parquet'):
                    parquet_files += 1
                if total_objects <= 10:  # Print first 10 objects
                    print(f"- {obj['Key']} (Size: {obj['Size']} bytes)")
    
    print(f"\nTotal objects found: {total_objects}")
    print(f"Parquet files found: {parquet_files}")
    
    if parquet_files == 0:
        print("\nWARNING: No parquet files found in the bucket!")
        print("This might explain why the visualization is not working.")
        
except Exception as e:
    print(f"\nError accessing S3 bucket: {str(e)}") 