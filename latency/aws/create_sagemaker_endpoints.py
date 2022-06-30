import boto3
from sagemaker import image_uris


def main():
    # TODO: add env file to load credentials
    aws_region = "us-east-1"
    sagemaker_client = boto3.client("sagemaker", region_name=aws_region)
    account_id = "721164497653"
    sagemaker_role = f"arn:aws:iam::{aws_region}:{account_id}:role/*"
