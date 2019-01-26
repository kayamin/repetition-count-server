from django.http import HttpResponse
import json
import numpy as np

# Create your views here.
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.csrf import csrf_exempt

from repetition_count_server.models import GlobalCounter, LocalCounter


@method_decorator(csrf_exempt, name='dispatch')
class Initializer(View):

    def get(self, request, *args, **kwargs):
        global_counter = GlobalCounter.objects.create()
        strides = global_counter.detector_strides
        init_array = np.zeros(global_counter.history_num)

        for stride in strides:
            LocalCounter.objects.create(global_counter=global_counter,
                                        interval=stride,
                                        label_array=init_array,
                                        count_array=init_array,
                                        ent_arr=init_array + 2,
                                        std_arr=init_array)

        data = json.dumps({"global_counter_id": global_counter.id})
        return HttpResponse(content=data, content_type='application/json', status=201)
