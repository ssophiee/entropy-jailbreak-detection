---
base_model: meta-llama/Llama-3.1-8B
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:meta-llama/Llama-3.1-8B
- lora
- transformers
---

# Model Card for Model ID

<!-- Provide a quick summary of what the model is/does. -->



## Model Details

### Model Description

import os
import torch
from dotenv import load_dotenv

load_dotenv()

DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device(DEVICE)

# MAX_LENGTH for tokenization
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "512"))

MAX_LAPLACE_BATCHES = int(os.getenv("MAX_LAPLACE_BATCHES", "40"))

# Model parameters
MODEL_NAME = "meta-llama/Llama-3.1-8B" # "Qwen/Qwen2.5-3B-Instruct"

# Data parameters
N_SAFE_TRAIN = 200  # Doubled from 100 for better Fisher information
N_BENIGN_TRAIN = 200  # Doubled from 100 for better Fisher information
N_TEST_PER_CATEGORY = 50

# Training parameters
BATCH_SIZE = 4  # Micro-batch size (effective batch = BATCH_SIZE * gradient_accumulation_steps)
EPOCHS = 5  # Increased from 3 for better convergence
MAX_LENGTH = 512
LEARNING_RATE = 2e-4 # 1e-2 # 1e-2  # Increased from 3e-4 for better Fisher information

# LoRA parameters - CRITICAL FOR MEMORY
LORA_RANK = 16  # Increased from 4 to 16 for better capacity (was too small!)
LORA_ALPHA = 32 #32  # Scale with rank (typically 2x rank)
LORA_DROPOUT = 0.1

# Bayesian parameters
LR_THRESHOLD = 1e-2  # Threshold for low-rank approximation
N_POSTERIOR_SAMPLES = 50  # Samples from predictive distribution
MAX_LAPLACE_BATCHES = 100  # Increased from 50 to use all training data (200 prompts with batch_size=2)
TEMPERATURE = 0.01  # Temperature for posterior sampling (use with sum-based Fisher)
PRIOR_PRECISION = 1 # 1e-4  # Prior precision for Laplace approximation (regularization for low-Fisher parameters)

# Directory to adapter checkpoints
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FINE_TUNED_MODEL_DIR = os.path.join(BASE_DIR, "saved_models")



- **Developed by:** [More Information Needed]
- **Funded by [optional]:** [More Information Needed]
- **Shared by [optional]:** [More Information Needed]
- **Model type:** [More Information Needed]
- **Language(s) (NLP):** [More Information Needed]
- **License:** [More Information Needed]
- **Finetuned from model [optional]:** [More Information Needed]

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Repository:** [More Information Needed]
- **Paper [optional]:** [More Information Needed]
- **Demo [optional]:** [More Information Needed]

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

[More Information Needed]

### Downstream Use [optional]

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

[More Information Needed]

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

[More Information Needed]

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

[More Information Needed]

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. More information needed for further recommendations.

## How to Get Started with the Model

Use the code below to get started with the model.

[More Information Needed]

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

[More Information Needed]

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing [optional]

[More Information Needed]


#### Training Hyperparameters

- **Training regime:** [More Information Needed] <!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->

#### Speeds, Sizes, Times [optional]

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

[More Information Needed]

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

[More Information Needed]

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

[More Information Needed]

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

[More Information Needed]

### Results

[More Information Needed]

#### Summary



## Model Examination [optional]

<!-- Relevant interpretability work for the model goes here -->

[More Information Needed]

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** [More Information Needed]
- **Hours used:** [More Information Needed]
- **Cloud Provider:** [More Information Needed]
- **Compute Region:** [More Information Needed]
- **Carbon Emitted:** [More Information Needed]

## Technical Specifications [optional]

### Model Architecture and Objective

[More Information Needed]

### Compute Infrastructure

[More Information Needed]

#### Hardware

[More Information Needed]

#### Software

[More Information Needed]

## Citation [optional]

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

[More Information Needed]

**APA:**

[More Information Needed]

## Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

[More Information Needed]

## More Information [optional]

[More Information Needed]

## Model Card Authors [optional]

[More Information Needed]

## Model Card Contact

[More Information Needed]
### Framework versions

- PEFT 0.18.1