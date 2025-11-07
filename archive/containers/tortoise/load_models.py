from tortoise import api

api.TextToSpeech(use_deepspeed=True, kv_cache=True)