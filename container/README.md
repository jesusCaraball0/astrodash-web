# Containerization for astrodash-web 

## Local Exploration Using Docker Compose

1. Install [Docker Compose](https://docs.docker.com/compose/install/)

1. Clone this repository:
   ```
   git clone https://github.com/scimma/astrodash-web.git
   ```

1. Change directories:
   ```
   cd astrodash-web/container
   ```

1. Start the services. If the container images have not been previously built and
   cached `compose` will build them.

   ```
   docker compose up -d
   
   ```

1. Monitor log files:
   ```
   docker compose logs -f
   ```

1. Browse to [http://localhost/](http://localhost)

1. Stop the services:
   ```
   docker compose down
   ```
