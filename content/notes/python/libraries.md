---
title: "Libraries"
date: "2026-10-27"
---

Because the Python Package Index (PyPI) hosts over 500,000 active third-party packages in addition to the standard built-in modules, listing *literally all* individual libraries item-by-item is physically impossible.

However, **all Python libraries fall into a well-defined hierarchy** structured by source type, domain, sub-domain, and primary ecosystem standards. Below is the functional hierarchy of the Python library ecosystem:

## 1\. Built-in / Standard Libraries (Python Runtime)

Libraries included out-of-the-box with standard Python installations.

- **Text Processing & String Formatting:** `string`, `re` (Regex), `difflib`, `unicodedata`
- **Data Types & Data Structures:** `datetime`, `collections`, `heapq`, `array`, `enum`, `dataclasses`, `types`
- **Numeric & Mathematical:** `math`, `cmath`, `decimal`, `fractions`, `random`, `statistics`
- **Functional Programming:** `itertools`, `functools`, `operator`
- **File System & Directory Handling:** `os`, `sys`, `pathlib`, `shutil`, `glob`, `tempfile`
- **Data Persistence & Compression:** `json`, `csv`, `sqlite3`, `pickle`, `zlib`, `zipfile`, `tarfile`
- **Networking & Internet Protocols:** `urllib`, `http`, `socket`, `ftplib`, `email`, `ssl`
- **Concurrency & Execution:** `threading`, `multiprocessing`, `asyncio`, `concurrent.futures`, `subprocess`
- **Development & Debugging:** `unittest`, `logging`, `traceback`, `pdb`, `timeit`, `cProfile`

## 2\. Data Science, AI & Machine Learning Ecosystem

Libraries built for numerical computation, statistical modeling, and artificial intelligence.

DataFlair

- **Core Numerical & Array Computing:**
	- `NumPy`, `SciPy`, `CuPy` (GPU computing)
		Medium
- **Data Manipulation & Analysis:**
	- `Pandas`, `Polars` (Rust-backed multi-threading), `Dask` (Distributed data processing), `Vaex`
- **Data Visualization:**
	- **Static/2D Plotting:** `Matplotlib`, `Seaborn`
		- **Interactive/Web Plotting:** `Plotly`, `Bokeh`, `Altair`, `Holoviews`
- **Machine Learning (Classical / Traditional):**
	- `scikit-learn`, `XGBoost`, `LightGBM`, `CatBoost`, `Statsmodels`
- **Deep Learning Frameworks:**
	- `PyTorch`, `TensorFlow`, `JAX`, `Keras`
- **Natural Language Processing (NLP) & LLMs:**
	- `transformers` (Hugging Face), `LangChain`, `LlamaIndex`, `spaCy`, `NLTK`, `Gensim`
- **Computer Vision & Image Processing:**
	- `OpenCV` (`opencv-python`), `Pillow` (PIL), `scikit-image`, `Albumentations`
- **Big Data & Distributed Computing:**
	- `PySpark`, `Ray`, `Modin`, `PyArrow`

## 3\. Web Development, APIs & Networking

Frameworks and utilities for backend, web APIs, networking, and web scraping.

OpenCV

- **Full-Stack Web Frameworks:**
	- `Django`, `Masonite`
- **Micro Frameworks & Asynchronous APIs:**
	- `FastAPI`, `Flask`, `Tornado`, `Bottle`, `Sanic`, `Aiohttp`
- **HTTP Clients & Requests:**
	- `requests`, `httpx`, `urllib3`
- **Web Scraping & Browser Automation:**
	- `BeautifulSoup4`, `Scrapy`, `Selenium`, `Playwright`, `Pyppeteer`
- **Databases & ORMs:**
	- `SQLAlchemy`, `SQLModel`, `Peewee`, `Tortoise-ORM`, `pymongo`, `psycopg2` / `asyncpg`

## 4\. Software Engineering, Testing & Dev Tools

Libraries for project management, testing, deployment, and environment control.

- **Testing & Quality Assurance:**
	- `pytest`, `unittest` (built-in), `hypothesis`, `coverage`, `tox`
- **Code Formatting & Linting:**
	- `black`, `flake8`, `ruff`, `mypy` (static typing), `isort`
- **CLI & Command-Line Interfaces:**
	- `Click`, `Typer`, `argparse` (built-in), `Rich`, `Prompt_toolkit`
- **Packaging & Dependency Management:**
	- `pip`, `poetry`, `hatch`, `flit`, `conda`, `uv`

## 5\. Desktop Application & GUI Frameworks

Libraries for graphical user interface desktop applications.

- **Native / Cross-Platform GUIs:**
	- `Tkinter` (built-in), `PyQt` / `PySide` (Qt bindings), `wxPython`, `Kivy` (Mobile/Touch)
- **Web-to-Desktop Apps:**
	- `Streamlit`, `Gradio`, `PySimpleGUI`, `NiceGUI`

## 6\. Systems, Hardware & Embedded Systems

Low-level computing, IoT, and hardware automation.

- **Embedded & Microcontrollers:** `MicroPython`, `CircuitPython`
- **Hardware Interfacing & Serial Communication:** `pyserial`, `RPi.GPIO`, `pyUSB`
- **System Administration & Automation:** `Paramiko` (SSH), `Ansible`, `Fabric`, `psutil`
