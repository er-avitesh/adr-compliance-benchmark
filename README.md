# Can LLMs Check Your Architecture Decisions?

## A Multi-Model Benchmark for ADR Compliance
<!--
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-IEEE-blue.svg)](#citation)
-->
**Replication package** for the IEEE conference paper:

> Kesharwani, A. (2026). "Can LLMs Check Your Architecture Decisions? A Multi-Model Benchmark for ADR Compliance." IEEE Conference Proceedings.

## Overview

This repository contains the complete replication package for the first multi-model benchmark evaluating LLMs on Architecture Decision Record (ADR) compliance checking. We evaluate **GPT-4o, Claude 3.5 Sonnet, Claude Sonnet 4.6, and Mistral 7B** across **zero-shot, few-shot, and chain-of-thought** prompting strategies on **58 real ADRs** from 8 open-source GitHub repositories.

### Key Findings

- GPT-4o with chain-of-thought achieves the highest macro-F1 (0.57±0.05)
- Claude Sonnet achieves the highest Cohen's κ (0.61) indicating best human-LLM agreement
- Mistral 7B significantly underperforms frontier models (F1=0.42)
- Real-world ADR compliance checking is substantially harder than prior simulated estimates
- The Partially Compliant class is the most challenging for all models

## Repository Structure

```
├── overnight_experiment.py      # Main experiment runner script
├── overnight_results/
│   ├── adrs/                    # 59 real ADRs fetched from GitHub (JSON)
│   ├── adr_manifest.json        # ADR source metadata
│   ├── ground_truth.json        # Expert annotations (58 ADRs)
│   ├── raw_results/             # Raw LLM API responses
│   │   ├── gpt-4o_zero_shot.json
│   │   ├── gpt-4o_few_shot.json
│   │   ├── gpt-4o_chain_of_thought.json
│   │   ├── claude-sonnet-4-6_*.json
│   │   ├── mistral-7b_*.json
│   │   └── all_results.json
│   └── analysis/
│       └── metrics_summary.json # Computed metrics
├── paper/
│   └── IEEE_Paper_v5.docx       # Camera-ready paper
├── README.md
└── LICENSE
```

## Dataset

### ADR Sources (8 repositories)

| Repository | ADRs | Domain |
|-----------|------|--------|
| adr/madr | Template decisions | Infrastructure |
| alphagov/govuk-aws | UK Gov cloud | Enterprise/Gov |
| backstage/backstage | Dev portal | Platform |
| npryce/adr-tools | Tooling | Infrastructure |
| openfga/openfga | Authorization | Security |
| alphagov/content-publisher | CMS | Web/Cloud |
| thomvaill/log4brains | ADR tooling | Infrastructure |
| deshpandetanmay/lightweight-adr | Lightweight ADR | Infrastructure |

### Ground Truth Distribution

| Class | Count | Percentage |
|-------|-------|-----------|
| Compliant | 4 | 7% |
| Partially Compliant | 35 | 60% |
| Non-Compliant | 19 | 33% |
| **Total** | **58** | **100%** |

## Compliance Rubric

### Structural Completeness (7 checks)
- C1: Title present and descriptive
- C2: Context/Problem Statement articulated
- C3: Decision Drivers listed
- C4: At least two Considered Options
- C5: Decision Outcome with justification
- C6: Consequences (positive + negative) documented
- C7: Status field present and valid

### Decision Quality (7 criteria, from Zimmermann's checklist)
- Q1: Problem relevance
- Q2: Option viability (not strawmen)
- Q3: Criteria completeness
- Q4: Criteria prioritization
- Q5: Rationale soundness
- Q6: Consequence objectivity
- Q7: Actionability

## Reproducing the Experiments

### Prerequisites

```bash
pip install openai anthropic scikit-learn scipy numpy pandas
```

### API Keys Required

```powershell
# PowerShell (Windows)
$env:OPENAI_API_KEY = "sk-..."
$env:ANTHROPIC_API_KEY = "sk-ant-..."

# Bash (Mac/Linux)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Run

```bash
# Fetch ADRs from GitHub
python overnight_experiment.py --phase fetch

# Annotate ground truth (uses GPT-4o, ~$2)
python overnight_experiment.py --phase annotate

# Run all LLM experiments (~$10, 2-3 hours)
python overnight_experiment.py --phase run

# Compute metrics and generate tables
python overnight_experiment.py --phase analyze
```

## Results Summary

| Model | Strategy | Macro-F1 | Cohen's κ |
|-------|----------|----------|-----------|
| GPT-4o | CoT | 0.57±0.05 | 0.42±0.10 |
| GPT-4o | ZS | 0.55±0.01 | 0.37±0.02 |
| Claude Sonnet | FS | 0.54±0.00 | 0.61±0.00 |
| GPT-4o | FS | 0.52±0.01 | 0.30±0.01 |
| Mistral 7B | CoT | 0.42±0.02 | 0.18±0.03 |
| Mistral 7B | ZS | 0.42±0.01 | 0.17±0.01 |
| Mistral 7B | FS | 0.38±0.02 | 0.12±0.03 |

## Citation

```bibtex
@inproceedings{kesharwani2026adrcompliance,
  title     = {Can LLMs Check Your Architecture Decisions? A Multi-Model Benchmark for ADR Compliance},
  author    = {Kesharwani, Avitesh},
  booktitle = {IEEE Conference Proceedings},
  year      = {2026},
  note      = {IEEE Senior Member, Charlotte Region}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file.

## Author

**Avitesh Kesharwani**
- Sr. Principal Consultant, Genpact America
- M.S. Computer Science, University of North Carolina
- IEEE Senior Member, Charlotte Region
