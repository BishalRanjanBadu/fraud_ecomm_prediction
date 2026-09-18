# Test fixtures (committed)

These files are **generated from the real bucket** and committed so that CI tests run with no cloud credentials:

```
python scripts/make_fixture.py --version <candidate or promoted version>
```

`make_fixture.py` refuses to write anything unless the service transform reproduces the notebooks' stage-06 rows
(tree: exact, linear: within 1e-12). Regenerate whenever a new model version is promoted; `FIXTURE_INFO.json`
records the version, library versions and the SHA-256 of every source object.
