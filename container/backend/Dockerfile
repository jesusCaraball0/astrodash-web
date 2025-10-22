# astrodash-web backend Dockerfile
#
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

ARG UID=1000
ARG USERNAME=dash
ARG DASH_BASE_DIR=/opt

RUN useradd --create-home --shell /bin/bash ${USERNAME} --uid ${UID}

WORKDIR ${DASH_BASE_DIR}

RUN mkdir -p ${DASH_BASE_DIR} && chown ${UID} -R ${DASH_BASE_DIR}

# Install Python dependencies
COPY prod_backend/requirements/prod.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /tmp/requirements.txt

USER ${USERNAME}

# Copy the DASH application into place.
COPY --chown=${USERNAME}:${USERNAME} prod_backend/app ${DASH_BASE_DIR}/app
