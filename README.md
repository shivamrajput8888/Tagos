# TAGOS

**AI-Powered Automated Content Tagging for Blogs**

TAGOS is a Flask-based web application that generates relevant tags for
blog content using a fine-tuned DistilBERT model. The production
deployment has been optimized to run on memory-constrained hosting by
replacing PyTorch inference with a quantized ONNX Runtime pipeline.

## Live Application

Production deployment: https://tagos-iwb9.onrender.com/

## Project Overview

TAGOS accepts blog content, processes it through a tokenizer and trained
multi-label classification model, and returns ranked tags with
confidence scores. The application also supports copying generated tags
and downloading a prediction report.

## Major Deployment Changes

The original application used a PyTorch `.pth` model at runtime. On
Render's 512 MB instance, the service repeatedly failed with an
out-of-memory error because the combined runtime stack and model
exceeded available memory.

The deployment pipeline was therefore redesigned:

1.  The trained PyTorch model was exported to ONNX format.
2.  The ONNX model was dynamically quantized to INT8.
3.  Model size was reduced from approximately **253.84 MB** to
    approximately **63.79 MB**.
4.  The quantized model was uploaded to a Hugging Face model repository
    instead of being stored directly in GitHub.
5.  Production inference was migrated from PyTorch to ONNX Runtime.
6.  PyTorch was removed from production dependencies.
7.  ONNX Runtime was configured for CPU inference with limited thread
    usage.
8.  The app now downloads the quantized ONNX model at startup only when
    the local model file is missing.
9.  Render successfully deploys the service on the available
    free-instance memory limit.

## Architecture

``` text
User Input
   |
   v
Flask Web Application
   |
   v
DistilBERT Tokenizer
   |
   v
Quantized INT8 ONNX Model
   |
   v
ONNX Runtime CPU Inference
   |
   v
Multi-label Predictions
   |
   v
Ranked Tags + Scores
   |
   v
Web UI / Downloadable Report
```

## Technology Stack

-   Python 3.11
-   Flask
-   Gunicorn
-   NumPy
-   ONNX Runtime
-   Hugging Face Transformers
-   Tokenizers
-   scikit-learn
-   Joblib
-   Requests
-   HTML/CSS/JavaScript
-   Render
-   Hugging Face Hub

## Key Files

### `app.py`

Main Flask application. Handles routes, form input, tag generation, and
web responses.

### `predict.py`

Production inference module. It now:

-   downloads the INT8 ONNX model when missing
-   loads the label encoder
-   loads the DistilBERT tokenizer
-   creates an ONNX Runtime CPU session
-   tokenizes incoming text
-   runs ONNX inference
-   applies sigmoid to logits
-   ranks predicted tags
-   returns top predictions

### `requirements.txt`

Production dependencies were reduced and updated to remove PyTorch.

Current production dependencies:

``` text
Flask==3.0.3
gunicorn==23.0.0
numpy==1.26.4
onnxruntime==1.20.1
transformers==4.43.3
tokenizers==0.19.1
joblib==1.4.2
scikit-learn==1.5.1
requests==2.32.3
```

### `convert_to_onnx.py`

Utility script used to convert the trained PyTorch TAGOS model into ONNX
format.

### `saved_model/`

Contains supporting model assets such as the label encoder and
tokenizer-related files. The large quantized ONNX model is hosted
externally and downloaded when required.

### `gunicorn.conf.py`

Gunicorn configuration used to run the Flask application in production.

## Model Optimization

### Original Model

``` text
Framework: PyTorch
Format: .pth
Approximate model size: 266 MB
Production issue: memory usage exceeded 512 MB
```

### Exported ONNX Model

``` text
Format: ONNX
Approximate size: 253.84 MB
```

### Quantized Production Model

``` text
Format: INT8 ONNX
Approximate size: 63.79 MB
Runtime: ONNX Runtime
Execution provider: CPUExecutionProvider
```

The INT8 model is roughly one quarter of the size of the unquantized
ONNX model.

## Model Hosting

The quantized production model is stored in the Hugging Face model
repository:

``` text
Shivacer8888/tagos-model
```

Expected production model filename:

``` text
tagos_model_int8.onnx
```

At application startup, `predict.py` checks whether the model exists
locally. If it is absent, the file is downloaded before the ONNX
inference session is initialized.

## Local Setup

### 1. Clone the repository

``` bash
git clone https://github.com/shivamrajput8888/Tagos.git
cd Tagos
```

### 2. Create a virtual environment

Windows:

``` bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

``` bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

### 4. Run the application

``` bash
python app.py
```

Then open the local address shown in the terminal.

## Production Deployment on Render

Recommended configuration:

``` text
Runtime: Python
Build Command: pip install -r requirements.txt
Start Command: gunicorn -c gunicorn.conf.py app:app
```

The deployed service uses Python 3.11 as configured by the repository.

During first startup, the application may need extra time to download
the INT8 ONNX model. Render free instances may also spin down after
inactivity, so the first request after an idle period can be slower.

## Deployment Troubleshooting History

### Issue 1: Out of Memory

Error:

``` text
Ran out of memory (used over 512MB) while running your code.
```

Cause:

-   PyTorch runtime
-   Transformers stack
-   DistilBERT model loading
-   limited Render instance memory

Resolution:

-   export model to ONNX
-   quantize model to INT8
-   replace PyTorch inference with ONNX Runtime
-   remove `torch` from production dependencies

### Issue 2: Missing ONNX Runtime

Error:

``` text
ModuleNotFoundError: No module named 'onnxruntime'
```

Cause:

The application code had already been changed to import ONNX Runtime,
but Render was still deploying an earlier commit/environment without the
updated dependency.

Resolution:

-   add `onnxruntime==1.20.1` to `requirements.txt`
-   deploy the latest commit
-   rebuild the service environment

## Successful Deployment Verification

The successful production logs include messages similar to:

``` text
Downloading INT8 ONNX model...
ONNX model downloaded successfully!
Loading label encoder...
Loading tokenizer...
Loading ONNX model...
ONNX model loaded successfully!
Your service is live
```

This confirms that the optimized model download, tokenizer setup, label
encoder loading, ONNX session creation, and Flask service startup
completed successfully.

## Important Notes

-   The deployment is operational, but prediction quality depends on the
    original training data, label distribution, and model calibration.
-   A successful deployment does not guarantee that every predicted tag
    is semantically ideal.
-   The current scoring/display logic should be reviewed separately if
    calibrated probabilities are required.
-   `bert_model.py` and other training-oriented PyTorch files may remain
    in the repository for training or historical purposes, but the
    production inference path should not import them.
-   Avoid re-adding `torch` to production requirements unless the
    deployment plan has sufficient memory.

## Repository

GitHub repository: https://github.com/shivamrajput8888/Tagos

## Current Status

**Deployment: Live**

**Inference runtime: ONNX Runtime**

**Production model: Quantized INT8 ONNX**

**Hosting: Render**
