# Deploying the demo

The Streamlit app is deployed from the GitHub repository. These steps need your
own Streamlit and GitHub accounts, so they are yours to run — but everything in
the repository is already set up for them.

## Streamlit Community Cloud (free)

1. Push this repository to GitHub (public is fine and keeps the free tier simple).
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **Create app** → **Deploy a public app from GitHub**.
4. Fill in:
   - **Repository:** `Adithyrao1/admet-ml`
   - **Branch:** `main`
   - **Main file path:** `app/streamlit_app.py`
5. Click **Deploy**.

The first build takes a few minutes while it installs RDKit and XGBoost. The app
then downloads the two datasets and trains its models on first load, which adds
about a minute to the very first page view. After that both are cached.

Your URL will look like `https://<app-name>.streamlit.app`. Put it at the top of
the README.

## Things that commonly go wrong

**Build fails on a dependency.** Community Cloud reads `requirements.txt` from
the repository root, which is already present and pinned. If a pin is
unavailable for the Python version Streamlit picks, relax that one line to a
minimum (`>=`) rather than removing it.

**App is slow on first load.** Expected: it trains two XGBoost models at
startup. `@st.cache_resource` keeps them warm afterwards. If cold starts bother
you, commit pre-trained models with `joblib` and load them instead — at the cost
of carrying binary files in the repository.

**App sleeps after inactivity.** Free apps idle out and take a few seconds to
wake. That is normal and nothing to fix.

**Memory limits.** The free tier allows about 1 GB. Both models together stay
well inside it, but adding a third endpoint or a much larger fingerprint would
need checking.

## Running it locally

```bash
pip install -r requirements.txt
pip install -e .
streamlit run app/streamlit_app.py
```

It opens at <http://localhost:8501>.
