# Golden payloads (committed)

`payloads.json` and `expected.json` are recorded **once per promoted model**, against the real model served
locally (runbook step "record golden payloads"):

```
python scripts/golden.py record --base-url http://127.0.0.1:8000
```

They are then checked, with the standard library only, at three points: local uvicorn, `docker run`, and
through the EKS load balancer (CI post-deploy job). CI's test job fails if these files are missing.
