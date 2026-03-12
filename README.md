# SmartHabit Dashboard Dockerizado

Este proyecto ha sido dockerizado para facilitar su ejecución y demostración (e.g., pitch de ventas) sin necesidad de configurar un servidor o dependencias locales complejas.

## Requisitos
- [Docker Desktop](https://www.docker.com/products/docker-desktop) instalado y en ejecución.

## Instrucciones para ejecutar

1. Abre una terminal en la carpeta raíz del proyecto (donde se encuentra este `README.md`).
2. Ejecuta el siguiente comando para construir e iniciar el contenedor en segundo plano:
   ```bash
   docker compose up -d
   ```
3. Abre tu navegador web favorito y visita la siguiente dirección:
   ```
   http://localhost
   ```

¡Ya deberías ver el dashboard funcionando correctamente!

## Para detener el proyecto
Cuando termines tu presentación, puedes detener el contenedor ejecutando:
```bash
docker compose down
```
