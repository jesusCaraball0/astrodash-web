# astrodash-web frontend Dockerfile
#
FROM node:22-trixie AS build

WORKDIR /app

COPY frontend/package.json ./

RUN npm install && npm ci

COPY frontend/src /app/src
COPY frontend/public /app/public
COPY frontend/tsconfig.json /app/tsconfig.json

ARG ASTRODASH_API_BASE_URL=http://localhost:8000
RUN REACT_APP_API_BASE_URL="${ASTRODASH_API_BASE_URL}" npm run build

FROM nginx:trixie

RUN rm -rf /usr/share/nginx/html/*

COPY --from=build /app/build /usr/share/nginx/html
