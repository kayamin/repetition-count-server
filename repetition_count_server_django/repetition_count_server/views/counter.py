from django.core.exceptions import ObjectDoesNotExist
from django.http import HttpResponse
import json
import numpy as np

# Create your views here.
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.csrf import csrf_exempt

from repetition_count_server.models import GlobalCounter, LocalCounter
from repetition_count_server.utils.state import State


@method_decorator(csrf_exempt, name='dispatch')
class Counter(View):

    def get(self, request, *args, **kwargs):
        # カウント結果を返す
        global_counter_id = int(request.GET.get("global_counter_id"))
        frame_count = int(request.GET.get("frame_count"))
        try:
            gc = GlobalCounter.objects.get(id=global_counter_id)  # type: GlobalCounter
        except ObjectDoesNotExist:
            return HttpResponse(content="Object Does not Exist", status=400)

        out_time = frame_count // 60

        content = dict(status="", global_counter=0)
        print(f"g_id: {global_counter_id}")
        print(f"state: {gc.cur_state}")
        print(f"time: {out_time - gc.in_time}")
              
        if (gc.cur_state == State.IN_REP) and (((out_time - gc.in_time) < 4) or (gc.global_counter < 5)):
            content["status"] = "new hypothesis"
            content["global_counter"] = gc.global_counter
            
        if (gc.cur_state == State.IN_REP) and ((out_time - gc.in_time) >= 4) and (gc.global_counter >= 5):
            content["status"] = "counting"
            content["global_counter"] = gc.global_counter
        
        if (gc.cur_state == State.COOLDOWN) and (gc.global_counter >= 5):
            content["status"] = "done"
            content["global_counter"] = gc.global_counter
        
        content = json.dumps(content)
        
        return HttpResponse(content=content, status=200)

    def post(self, request, *args, **kwargs):

        data = json.loads(request.body)
        global_counter_id = data["global_counter_id"]
        interval = data["interval"]
        cur_std = data["cur_std"]
        frame_count = data["frame_count"]

        # RepCount モデルの出力を取得
        labels = [3, 4, 5, 6, 7, 8, 9, 10]
        p_y = [data[str(label)] for label in labels]
        output_label = labels[np.argmax(p_y)]
        print(f"g_id: {global_counter_id}")
        print(f"label: {output_label}")
        print(f"p_y: {p_y}")

        try:
            gc = GlobalCounter.objects.get(id=global_counter_id)  # type: GlobalCounter
            lc = LocalCounter.objects.get(global_counter_id=global_counter_id, interval=interval)  # type: LocalCounter
        except ObjectDoesNotExist:
            return HttpResponse(content="Object Does not Exist", status=400)

        gc.in_frame_num = frame_count
        print(f"cur_state: {gc.cur_state}")
        print(f"global_counter: {gc.global_counter}")

        def _do_local_count(initial: bool) -> float:
            cur_entropy = - (p_y * np.log(p_y)).sum()
            print(cur_entropy)
            # count
            lc.label_array = np.delete(lc.label_array, 0, axis=0)
            lc.label_array = np.insert(lc.label_array, -1, output_label, axis=0)
            # take median of the last 4
            med_out_label = np.ceil(np.median(lc.label_array[gc.history_num - 4:gc.history_num]))
            med_out_label = med_out_label.astype('int32')
            if initial:
                lc.rep_count = 20 // med_out_label
                lc.frame_residue = 20 % med_out_label
            else:
                lc.frame_residue += 1
                if lc.frame_residue >= med_out_label:
                    lc.rep_count += 1
                    lc.frame_residue = 0

            return cur_entropy

        # local count を移植，cur
        cur_entropy = 0
        if gc.cur_state == State.NO_REP:
            cur_entropy = _do_local_count(True)
        if (gc.cur_state == State.IN_REP) and (gc.winner_stride == lc.interval):
            cur_entropy = _do_local_count(False)
        if gc.cur_state == State.COOLDOWN:
            cur_entropy = _do_local_count(True)

        # common to all states
        if cur_std < gc.static_th:
            cur_entropy = 2

        lc.count_array = np.delete(lc.count_array, 0, axis=0)
        lc.count_array = np.insert(lc.count_array, gc.history_num - 1, lc.rep_count, axis=0)
        lc.ent_arr = np.delete(lc.ent_arr, 0, axis=0)
        lc.ent_arr = np.insert(lc.ent_arr, gc.history_num - 1, cur_entropy, axis=0)
        lc.std_arr = np.delete(lc.std_arr, 0, axis=0)
        lc.std_arr = np.insert(lc.std_arr, gc.history_num - 1, cur_std, axis=0)

        st_std = np.median(lc.std_arr)
        st_entropy = np.median(lc.ent_arr)

        if gc.cur_state == State.NO_REP:
            # 早いもの勝ちで用いるカウンターを決める. 異なるself.stride_number で複数存在するCounter インスタンスの中で下記 if 文の条件
            # を満たしたものが， グローバル変数 winner_stride に自身の self.stride_numebr をセットし，以降のカウンティングを行う
            # if we see good condition for rep, take the counting and move to rep state
            if (st_std > gc.norep_std_th) and (st_entropy < gc.norep_ent_th):
                # start counting!
                gc.actions_counter += 1
                gc.cur_state = State.IN_REP
                gc.global_counter = lc.rep_count
                gc.winner_stride = lc.interval
                gc.in_time = gc.in_frame_num // 60

        elif (gc.cur_state == State.IN_REP) and (gc.winner_stride == lc.interval):
            lastSixSorted = np.sort(lc.ent_arr[gc.history_num - 8:gc.history_num])
            # keep counting while entropy is low enough
            if (((st_std > gc.inrep_std_th) and (st_entropy < gc.inrep_ent_th)) or (
                    lastSixSorted[1] < gc.lastsix_ent_th)):
                # continue counting
                gc.global_counter = lc.rep_count
            else:
                gc.out_time = gc.in_frame_num // 60
                if ((gc.out_time - gc.in_time) < 4) or (lc.rep_count < 5):
                    # fast recovery mechanism, start over
                    # gc.actions_counter -= 1
                    # gc.global_counter = 0
                    # gc.cur_state = State.NO_REP
                    print('fast recovery applied !!')
                else:
                    # rewind redundant count mechanism
                    # find how many frames pass since we have low entropy
                    frames_pass = 0
                    reversed_ent = lc.ent_arr[::-1]
                    for cent in reversed_ent:
                        if cent > gc.inrep_ent_th:
                            frames_pass += 1
                        else:
                            break
                    # calc if and how many global count to rewind
                    reversed_cnt = lc.count_array[::-1]
                    frames_pass = min(frames_pass, reversed_cnt.shape[0] - 1)
                    new_counter = reversed_cnt[frames_pass]
                    print('counting rewinded by {}'.format(gc.global_counter - new_counter))
                    gc.global_counter = new_counter
                    # stop counting, move to cooldown
                    gc.cur_state = State.COOLDOWN
                    # init cooldown counter
                    gc.cooldown_in_time = gc.in_frame_num // 60

        elif gc.cur_state == State.COOLDOWN:
            gc.cooldown_out_time = gc.in_frame_num // 60
            if (gc.cooldown_out_time - gc.cooldown_in_time) > 4:
                gc.global_counter = 0
                gc.cur_state = State.NO_REP

        # GlobalCounter, LocalCounter の値を保存
        gc.save()
        lc.save()

        
        print(data['global_counter_id'])

        return HttpResponse("Posted")
