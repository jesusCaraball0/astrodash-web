# Developer documentation

## Data file initialization

After cloning the source code repo, install the data files stored in Git LFS:

```bash
git lfs pull
```

### Initial data S3 upload and manifest generation

Prepare the S3 bucket for storing the version-controlled DASH initial data objects:

```bash
# S3 endpoint configuration using MinIO CLI "mc"
#
$ mc alias ls js-blast
js-blast
  URL       : https://js2.jetstream-cloud.org:8001
  AccessKey : ******
  SecretKey : ******
  API       : s3v4
  Path      : auto

# Enable bucket versioning
#
mc version enable js-blast/astrodash
mc version info js-blast/astrodash

# Upload data files from local Git LFS clone
#
mc mirror ${GIT_CLONE_PATH}/data/ js-blast/dash/init_data/

# Set the access credential env vars
export S3_ACCESS_KEY_ID_INIT="****"
export S3_SECRET_ACCESS_KEY_INIT="****"

# Install dependencies in a virtual environment
cd ${GIT_CLONE_PATH}/scripts
python -m venv .venv
source .venv/bin/activate
pip install minio

# Run the generator
#
$ python generate_manifest.py

Creating a MinIO client...
Listing objects in "s3://js2.jetstream-cloud.org:8001/dash/init_data/"...
Writing manifest file "../astrodash-data.json"...
Manifest generation complete.
```

## Run DASH locally in Docker Compose

Build and launch the application using:

```bash
docker compose up -d --build
# optionally follow container logs:
docker compose logs -f
```

Open your browser to http://localhost:8888 to access the web app.

## Build Frontend with Custom Backend API Base URL

The default backend API base URL used by the frontend is
http://localhost:8000.

To build the static frontend with a custom backend API base URL:

```bash
export ASTRODASH_API_BASE_URL=https://api.dash.ncsa.illinois.edu
docker compose up -d --build
```

Alternatively, create a Docker Compose envrionment file at `/container/.env` with custom values; for example:

```bash
ASTRODASH_API_BASE_URL=http://dev.dash.ncsa.illinois.edu
# IMAGE_FRONTEND=hub.ncsa.illinois.edu/blast/astrodash-web-frontend:dev2
```

## Deploying to production

Deploying to production involves overriding some chart values like so:

```bash
$ helm upgrade --install --create-namespace --namespace astrodash astrodash . -f values.yaml -f values.prod.yaml 
```

This assumes that the `prod` image tag referenced by `values.prod.yaml` was built in `docker compose build` after setting in `.env` as shown below.

```bash
ASTRODASH_API_BASE_URL=https://dash.scimma.org
IMAGE_FRONTEND=hub.ncsa.illinois.edu/blast/astrodash-web-frontend:prod
```
