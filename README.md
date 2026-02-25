# Energy–Accuracy Pareto Frontiers for NLP Tasks

This project investigates accuracy–efficiency tradeoffs in pretrained NLP models by treating energy and latency as first-class evaluation metrics alongside task performance.

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
gcloud compute ssh <VM_NAME> --zone us-central1-a
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

## Phase 1: Establish Baseline

We fine-tune `bert-base-uncased` on SST-2 (GLUE) and measure:

- Validation accuracy
- Inference latency (ms per example)
- Energy consumption (J per example) via NVML power sampling
- Peak GPU memory usage

Specifically, we established a controlled baseline using:

- Model: `bert-base-uncased`
- Dataset: `SST-2` (`GLUE`)
- Max sequence length: `128`
- Batch size: `32`
- Epochs: `2`
- Learning rate: `3e-5`
- Weight decay: `0.01`
- Warmup ratio: `0.06`
- Seed: `42`

We obtained the following results:

- Validation accuracy: `0.9300`
- Latency: `~3.18 ms / example`
- Energy: `~0.231 J / example`
- Energy per correct prediction: `~0.249 J`
- Peak GPU memory: `~720 MB`

You may replicate this baseline by running Phase 1 on your VM instance with the following command:

```
python -m src.run_phase_1
```

Once your baseline run completes, all outputs are stored under:

```
runs/<timestamp>_phase_1_baseline/
```

This directory will include:

- `metrics.json`
- `power_trace.csv`
- `pareto.csv`
- `pareto_frontier.png`
- `best_model/`

You may download that output to your local machine as follows:

```
gcloud compute scp --recurse \
  <VM_NAME>:~/nlpfrontiers-research/runs/<run_id> \
  ./outputs \
  --zone <ZONE>
```

This baseline configuration is frozen and used for all subsequent efficiency comparisons in Phase 2 to ensure apples-to-apples analysis.

## Phase 2: Run Experiments

In Phase 2, we want to evaluate a set of efficiency techniques that target different sources of redundancy: (i) max sequence length reduction, (ii) layer reduction, and (iii) quantization.

### Max Sequence Length

You may evaluate max sequence length reduction (from `128` to `16`) with the following command:

```
python -m src.run_phase_2_max_sequence_length \
  --baseline_model_directory runs/<run_id>/best_model \
  --sequence_lengths 128 96 64 48 40 32 24 16
```

Then, you may download that output to your local machine as follows:

```
gcloud compute scp --recurse \
  nlpfrontiers-vm:~/nlpfrontiers-research/runs/<run_id> \
  ./phase_2_max_sequence_length_sweep \
  --zone us-central1-a \
  --project final-project-488320
```

This will include:

- `pareto.csv`
- `energy_accuracy_max_sequence_length.png`
- `energy_latency_max_sequence_length.png`

### Layers

You may evaluate layers reduction (from `12` to `2`) with the following command:

```
python -m src.run_phase_2_layers \
  --baseline_model_directory runs/<run_id>/best_model \
  --num_layers 12 10 8 6 4 2 \
  --max_sequence_length 128
```

Then, you may download that output to your local machine as follows:

```
gcloud compute scp --recurse \
  nlpfrontiers-vm:~/nlpfrontiers-research/runs/<run_id> \
  ./phase_2_layers_sweep \
  --zone us-central1-a \
  --project final-project-488320
```

This will include:

- `pareto.csv`
- `energy_accuracy_layers.png`
- `energy_latency_layers.png`

### Precision

```
python -m src.run_phase_2_precision \
  --baseline_model_directory runs/<run_id>/best_model \
  --precisions fp32 fp16 fp8 \
  --max_sequence_length 128 \
  --skip_failed_precisions
```

Then, you may download that output to your local machine as follows:

```
gcloud compute scp --recurse \
  nlpfrontiers-vm:~/nlpfrontiers-research/runs/<run_id> \
  ./phase_2_precision_sweep \
  --zone us-central1-a \
  --project final-project-488320
```

This will include:

- `pareto.csv`
- `energy_accuracy_precision.png`
- `energy_latency_precision.png`