from whisperspeech.pipeline import Pipeline

Pipeline(
    optimize=False,
    torch_compile=False,
    s2a_ref="collabora/whisperspeech:s2a-v1.95-medium-7lang.model",
    t2s_ref="collabora/whisperspeech:t2s-v1.95-medium-7lang.model"
)