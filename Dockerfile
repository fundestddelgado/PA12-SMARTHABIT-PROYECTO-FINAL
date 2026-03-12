FROM nginx:alpine

# Copiar el archivo principal como index.html para que Nginx lo sirva por defecto en la raíz
COPY Dashboard/dashboard_multihorizonte.html /usr/share/nginx/html/index.html

# Copiar el directorio estático (CSS, JS, JSON)
COPY Dashboard/static /usr/share/nginx/html/static

EXPOSE 80
