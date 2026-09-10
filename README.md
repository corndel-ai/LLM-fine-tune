# Chemistry LLM domain adaptation

[![Open the GPU notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/corndel-ai/LLM-fine-tune/blob/main/chemistry_llm_colab.ipynb)

This repository contains a teaching notebook for domain-adaptive pretraining
(DAPT) with LoRA on a small collection of OECD chemistry test guidelines.

## Run it in Google Colab

1. Click the **Open in Colab** badge above.
2. Select a GPU runtime (**Runtime > Change runtime type > T4 GPU**, or another
   available NVIDIA GPU).
3. Choose **Runtime > Run all**.

The notebook downloads the PDFs directly from their raw GitHub URLs. It needs no
repository clone, Google Drive mount, Hugging Face token, manual upload, or
interactive prompt. It uses a revision-pinned Phi-3 Mini model in 4-bit
precision and trains a LoRA adapter. The final adapter ZIP is created in
Colab's temporary `/content` storage; downloading it is optional and disabled
by default.

Colab runtimes are temporary and have dynamic resource limits:
https://research.google.com/colaboratory/faq.html

## What the exercise demonstrates

- page-aware PDF extraction and conservative cleaning;
- document-level train/validation splitting to reduce leakage;
- causal-language-model token construction without mixing documents;
- 4-bit quantization plus LoRA over linear layers;
- checkpointing, evaluation, early stopping, and comparison with a baseline;
- saving a reproducible adapter bundle and run metadata.

This is a short educational run, not a production chemistry assistant. A lower
validation loss or a more domain-styled answer does not establish factual
correctness or safety.

## Materials and licensing

Repository code and notebooks are licensed under the [MIT License](LICENSE).
The included OECD PDFs are separate third-party materials and are **not**
covered by the MIT License. See [OECD_MATERIALS.md](OECD_MATERIALS.md) for each
document's official DOI and the applicable OECD terms. This independent
teaching repository is not approved or endorsed by the OECD.

## Local regression check

The generation helper can be checked on CPU with a tiny randomly initialised
model, without downloading Phi-3 or running training. The test executes the
helper directly from the notebook and checks chat inputs and response slicing.
In a separate virtual environment, run:

```sh
python -m pip install transformers==5.16.1 torch
python -m unittest discover -s tests -v
```

This check does not replace a complete GPU run in Colab.
