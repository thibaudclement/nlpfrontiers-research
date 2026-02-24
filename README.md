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
```

Alternatively, you may set your Hugging Face access token as an environment variable with:

```
export HF_TOKEN=<YOUR_HF_ACCESS_TOKEN>
```

## Phase 1: Establish Baseline

We fine-tune `bert-base-uncased` on SST-2 (GLUE) and measure:

- Validation accuracy
- Inference latency (ms per example)
- Energy consumption (J per example) via NVML power sampling
- Peak GPU memory usage

You may replicate the baseline by running Phase 1 on your VM instance with the following command:

```
python -m src.run_phase_1
```

Once your baseline run completes, outputs are stored under:

```
runs/<timestamp>_phase_1_baseline/
```

You can download that output to your local machine as follows:

```
gcloud compute scp --recurse \
  <VM_NAME>:~/nlpfrontiers-research/runs/<run_id> \
  ./outputs \
  --zone <ZONE>
```