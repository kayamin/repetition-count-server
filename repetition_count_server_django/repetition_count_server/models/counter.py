from django.db import models
from picklefield import PickledObjectField

from repetition_count_server.utils.state import State


class GlobalCounter(models.Model):
    # グローバル変数を格納しておくモデル

    # 固定パラメータ
    detector_strides = PickledObjectField(default=(5, 7, 9))
    static_th = models.FloatField(default=10)
    norep_std_th = models.FloatField(default=13)
    norep_ent_th = models.FloatField(default=1.0)
    inrep_std_th = models.FloatField(default=13)
    inrep_ent_th = models.FloatField(default=1.1)
    lastsix_ent_th = models.FloatField(default=1.1)
    history_num = models.IntegerField(default=9)

    # アップデートするパラメータ
    in_time = models.IntegerField(default=0)
    out_time = models.IntegerField(default=0)
    cooldown_in_time = models.IntegerField(default=0)
    cooldown_out_time = models.IntegerField(default=0)
    global_counter = models.IntegerField(default=0)
    winner_stride = models.IntegerField(default=0)
    cur_state = models.IntegerField(default=State.NO_REP)
    in_frame_num = models.IntegerField(default=-1)
    actions_counter = models.IntegerField(default=0)

    def update(self, id: int, params: dict):
        GlobalCounter.objects.filter(id=id).update(**params)


class LocalCounter(models.Model):
    # 各カウンターのメンバ変数を格納しておくモデル
    global_counter = models.ForeignKey(GlobalCounter, on_delete=models.CASCADE)
    interval = models.IntegerField()

    rep_count = models.IntegerField(default=0)
    frame_residue = models.IntegerField(default=0)

    count_array = PickledObjectField()
    ent_arr = PickledObjectField()
    std_arr = PickledObjectField()

    label_array = PickledObjectField()

    def update(self, global_counter: GlobalCounter, interval: int, params: dict):
        LocalCounter.objects.filter(global_counter_id=global_counter.id, interval=interval).update(**params)
