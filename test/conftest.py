from hypothesis import settings

# Torch call latency varies (cold start, thread pool), so the default 200 ms
# per-example deadline produces flaky failures.
settings.register_profile("default", deadline=None)
settings.load_profile("default")
