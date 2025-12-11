from minio import Minio
import os
from pathlib import Path
import json


'''
Generate a JSON-formatted manifest of the latest versions of S3 objects.

Set S3_ACCESS_KEY_ID_INIT and S3_SECRET_ACCESS_KEY_INIT in the env vars.
'''
endpoint = os.getenv("S3_ENDPOINT_URL_INIT", 'js2.jetstream-cloud.org:8001')
bucket = os.getenv("S3_BUCKET_INIT", 'astrodash')
base_path = os.getenv("S3_BASE_PATH_INIT", "init_data/")
print('Creating a MinIO client...')
client = Minio(
    endpoint=endpoint,
    access_key=os.getenv('S3_ACCESS_KEY_ID_INIT'),
    secret_key=os.getenv('S3_SECRET_ACCESS_KEY_INIT'),
    secure=True,
    # region='',
)

print(f'Listing objects in "s3://{os.path.join(endpoint, bucket, base_path)}"...')
objects = client.list_objects(
    bucket_name=bucket,
    prefix=base_path,
    include_version=True,
    recursive=True,
)

# Collect metadata for each file
file_info = []
for obj in [obj for obj in objects]:
    info = {
        'path': obj.object_name.replace(base_path, ''),
        'version_id': obj.version_id,
        'etag': obj.etag,
        'size': obj.size,
    }
    if obj.is_latest.lower() == "true":
        file_info.append(info)

manifest_filepath = os.path.join(Path(__file__).resolve().parent.parent, 'astrodash-data.json')
print(f'Writing manifest file "{manifest_filepath}"...')
with open(manifest_filepath, 'w') as fh:
    json.dump(file_info, fh, indent=2)

print('Manifest generation complete.')
