"""
ASGI config for ImgSearchWebApp project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/6.1/howto/deployment/asgi/
"""

import os
from django.core.asgi import get_asgi_application
from starlette.staticfiles import StaticFiles

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ImgSearchWebApp.settings')
django_app = get_asgi_application()

from api.fastapi_app import fastapi_app

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
static_dir = os.path.join(BASE_DIR, 'search_app', 'static')
media_dir = os.path.join(BASE_DIR, 'media')

static_app = StaticFiles(directory=static_dir)
media_app = StaticFiles(directory=media_dir)

class CombinedASGIApp:
    """
    Routes /api, /docs, /openapi.json to FastAPI;
    Routes /static to static files;
    Routes /media to media files;
    Routes all other requests (including templates and admin) to Django.
    """
    def __init__(self, django_application, fastapi_application, static_application, media_application):
        self.django_app = django_application
        self.fastapi_app = fastapi_application
        self.static_app = static_application
        self.media_app = media_application

    async def __call__(self, scope, receive, send):
        path = scope.get('path', '')
        if scope.get('type') == 'http':
            if path.startswith('/static/'):
                # Strip prefix for Starlette StaticFiles if needed or route directly
                scope['path'] = path[len('/static'):]
                await self.static_app(scope, receive, send)
                return
            elif path.startswith('/media/'):
                scope['path'] = path[len('/media'):]
                await self.media_app(scope, receive, send)
                return
            elif (
                path.startswith('/api/') or 
                path == '/api' or 
                path.startswith('/docs') or 
                path.startswith('/openapi.json') or 
                path.startswith('/redoc')
            ):
                await self.fastapi_app(scope, receive, send)
                return

        await self.django_app(scope, receive, send)

application = CombinedASGIApp(django_app, fastapi_app, static_app, media_app)
