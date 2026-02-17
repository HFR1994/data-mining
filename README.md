# Data Mining Project

## Deployment

### Option A — Run it in Collab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12CafL7UZgVMG88HkZv0fkKtGdUr3kCxs?usp=sharing)

### Option B — Devcontainer (Recommended)

This project includes a devcontainer that provides a fully isolated development environment with all required tools preinstalled. Devpod automatically configures the workspace so you can start coding immediately without installing dependencies on your local machine.

#### How to use it

1. Make sure you have **Docker** or **Podman** enabled.
2. Install **[Devpod](https://devpod.sh)**.
3. Configure a provider as described **[here](https://devpod.sh/docs/managing-providers/add-provider)**.
4. Select the repository, choose your preferred IDE, and provide a workspace name.

The configuration will automatically create a development-ready environment with everything required to run the app.

---

### Option C — Manual Setup

1. Install **Python 3.14**.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Project

### Run the Shiny app (from the project root):

```bash
shiny run --launch-browser app.py
```

Access it at: [http://localhost:8000](http://localhost:8000)

### Run the Jupyter notebook:

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

Access it at: [http://localhost:8888](http://localhost:8888)
(The token appears in the terminal logs.)

---

This study explores the application of clustering algorithms to identify COVID-19 patient susceptibility groups based on symptoms, vital signs, 
and demographic information. Using data from two hospital datasets com prising 26,237 patients, we compare the performance of BIRCH, DBSCAN, 
and K-Means algorithms. Initial analysis with BIRCH on the full feature set achieved a silhouette score of 0.977, while dimension reduction techniques 
identified fatigue/malaise, sore throat, headache, age, and a combined oxygen-fever metric as key discriminative features. Results are presented at the end of the report based on the clustering results. 