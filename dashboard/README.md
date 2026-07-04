# Legacy Dashboard Directory

This directory is retained as a legacy/experimental dashboard path. It is not
the active portfolio dashboard.

The active Streamlit entry point is:

```bash
poetry run streamlit run app/streamlit_app.py
```

The repository target is:

```bash
make dashboard
```

New promoted CxG portfolio work should be added to `app/streamlit_app.py`, not
to `dashboard/app.py` or `dashboard/components/data_loader.py`. The legacy app
contains older CxG presentation assumptions and should not be used as the
primary review surface.
