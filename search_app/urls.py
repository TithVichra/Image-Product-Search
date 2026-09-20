from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/status', views.api_status, name='api_status'),
    path('api/search/text', views.api_search_text, name='api_search_text'),
    path('api/search/image', views.api_search_image, name='api_search_image'),
    path('api/detect', views.api_detect, name='api_detect'),
    path('api/index', views.api_index, name='api_index'),
]
