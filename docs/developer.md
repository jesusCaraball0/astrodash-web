# Developer documentation

## Data file initialization

After cloning the source code repo, install the data files stored in Git LFS:

```bash
git lfs pull
```

## Run DASH locally in Docker Compose

Build and launch the application using:

```bash
docker compose up -d --build
# optionally follow container logs:
docker compose logs -f
```

Open your browser to http://localhost:8888 to access the web app.
