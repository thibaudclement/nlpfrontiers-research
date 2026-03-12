# Energy–Accuracy Trade-offs in Transformer-Based NLP Models: A Unified Benchmarking Study

This repository contains the code and experimental framework for _Energy–Accuracy Trade-offs in Transformer-Based NLP Models: A Unified Benchmarking Study_, a Stanford [CS224N: Natural Language Processing with Deep Learning (Winter 2026)](https://web.stanford.edu/class/cs224n/) final project by Thibaud Clement. An online version of the final report is available [here](https://research.nlpfrontiers.com/).

## Abstract

Modern natural language processing (NLP) systems rely on transformer-based architectures that achieve strong predictive performance but require substantial computational resources. As models continue to scale, understanding the trade-offs between accuracy and energy consumption has become increasingly important. However, prior work often evaluates efficiency techniques in isolation, making it difficult to systematically evaluate architectural compression methods and inference-time optimizations under consistent experimental conditions. In this project, we investigate energy–accuracy trade-offs in transformer-based NLP models using a controlled benchmarking framework that measures GPU power consumption during both training and inference. Using the [SQuAD v2](https://huggingface.co/datasets/rajpurkar/squad_v2) question answering benchmark, we evaluate several BERT-derived architectural variants alongside inference-time optimizations within a unified experimental setting. Precision reduction emerges as the most effective efficiency strategy, reducing inference energy by up to 75% while preserving predictive performance. More broadly, different techniques affect different parts of the computational pipeline: architectural modifications such as pruning and layer freezing primarily reduce training energy, whereas inference-time techniques shift the deployment-time energy–accuracy frontier. Sequence length truncation produces a smooth Pareto frontier between energy consumption and performance, allowing moderate energy reductions with limited accuracy degradation. Finally, the most energy-efficient strategy depends on the expected workload: architectures with lower training energy are preferable when inference volume is small, whereas configurations with lower per-example inference energy dominate at large deployment scales.

## Repository structure

The repository contains two versions of the experimental framework:

### V1 (`/v1`) — Initial framework

The initial phase of the project focused on exploring inference-time efficiency techniques for the `bert-base-uncased` model. Experiments were conducted on several NLP benchmarks, including:

- SST-2 (sentiment classification)
- QQP (duplicate question detection)
- SQuAD v1.1 (extractive question answering)

This framework evaluated several inference-time reduction strategies:

- sequence length truncation
- encoder depth variation (retaining the first *k* layers)
- reduced floating-point precision (e.g., FP16)
- combinations of these techniques

### Final framework (root directory)

The final version of the project significantly expands the experimental design. It introduces a unified benchmarking framework that evaluates both architectural efficiency strategies (e.g., pruning, layer freezing, distilled models), and inference-time optimizations (precision reduction, sequence truncation, token pruning, early exit) while directly measuring GPU energy consumption during training and inference. The experiments reported in the final project use this framework and focus primarily on the SQuAD v2 dataset.

## Environment Setup

We used, and recommend for reproducibility, a VM instance in Google Cloud Platform, as follows:

- Region: `us-central1`
- Zone: `us-central1-a`
- Machine family: `GPU`
- Operating system: `Ubuntu 22.04` or `Ubuntu 24.04`
- Architecture : `x86_64`
- Series: `NVIDIA L4` (`g2-standard-4`)
- Driver-supported CUDA: `13.1`
- Provisioning: `Standard`
- Disk: `100GB` (`balanced persistent`)
- Python: `3.12`
- PyTorch: `cu121 build`

We use `<VM_NAME>` to interact with that VM instance in the rest of this project.

You may SSH into the VM via:

```
gcloud compute ssh <VM_NAME> --zone <ZONE>
```

We recommend the following steps for installation purposes:

```
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

For faster downloads, we recommend to include a Hugging Face access token in a `.env` file, as follows:

```
HF_TOKEN=<YOUR_HF_ACCESS_TOKEN>
HUGGINGFACE_HUB_TOKEN=<YOUR_HF_ACCESS_TOKEN>
```

Alternatively, you may set your Hugging Face access token as an environment variable in the via the VM command line with:

```
echo 'export HF_TOKEN=<YOUR_HF_ACCESS_TOKEN>' >> ~/.bashrc
echo 'export HUGGINGFACE_HUB_TOKEN=<YOUR_HF_ACCESS_TOKEN>' >> ~/.bashrc
source ~/.bashrc
```

## Running experiments

To train a model, use:

```
python -m src.<CLI_MODULE_NAME> \
--dataset-config-path configs/datasets/<CONFIG_FILE> \
--model-config-path configs/models/<CONFIG_FILE> \
--training-config-path configs/training/<CONFIG_FILE>
```

To run a sweep, use:

```
python -m src.<CLI_MODULE_NAME> \
  --dataset-config-path configs/datasets/<CONFIG_FILE> \
  --model-config-path configs/models/<CONFIG_FILE> \
  --inference-config-path configs/inference/<CONFIG_FILE> \
  --checkpoint-path runs/<RUN_ID>/huggingface_trainer/<CHECKPOINT_ID> \
  --model-label <MODEL_ID>
```

All experiment outputs are timestamped and stored in `runs/`.

To generate a plot, use:

```
python -m src.evaluation.<MODULE_NAME> \
--outputs-root <DESTINATION_ROOT> \
--output-path <DESTINATION_ROOT>/<DESTINATION_ROOT>/<FILE_NAME> \
--plot-title "<PLOT_TITLE>" \
--color-map <COLORMAP_ID>
```

## Acknowledgments

The author gratefully acknowledges Professors [Diyi Yang](https://cs.stanford.edu/~diyiy/) and [Yejin Choi](https://yejinc.github.io/), as well as Head TA [Julie Kallini](https://juliekallini.com/) and the CS224N teaching team, for their instruction, guidance, and inspiration throughout the course. Special thanks are also due to the Final Project Mentor, [David Anugraha](https://davidanugraha.github.io/), for his thoughtful feedback, support, and guidance during the development of this project.