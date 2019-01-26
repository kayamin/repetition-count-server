
from django.urls import path

from repetition_count_server.views import hello
from repetition_count_server.views import Counter, Initializer 

urlpatterns = [
    path('counter/', Counter.as_view()),
    path('init/', Initializer.as_view())
]   