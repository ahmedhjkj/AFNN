# Adaptive Fractal Neural Networks (AFNN)
## A Dynamic Architecture for Learning Computational Structure

**Technical Whitepaper Draft**  
**Author:** Ahmed Al-Amin  
**Date:** September 2026  
**Project:** [AFNN — Adaptive Fractal Neural Network](https://github.com/ahmedhjkj/AFNN)

> **Research status.** AFNN is a working prototype and a preliminary research project. The current experiment demonstrates trainability, but it does not yet establish superiority over CNNs, ResNets, FractalNet, or other existing architectures. Controlled baseline experiments are planned as the next stage.

## Abstract

Most neural networks are designed with a fixed computational structure before training begins. Training then changes the numerical values of the weights inside that structure, but not the structure itself. AFNN investigates a different organization: a recursively constructed, fractal-like network whose branch structure can change during training.

AFNN combines three ideas. First, its computational blocks are built recursively, creating a self-similar organization of branches. Second, branch outputs are combined with learned softmax weights rather than a fixed averaging rule. Third, a structural control loop monitors training behavior and can add, prune, or rejuvenate branches subject to explicit limits.

In the reported proof-of-concept experiment, AFNN classified images from 94 classes and reached **74.1% validation accuracy** using approximately 56,000 images resized to 48 × 48 pixels. The model used a reported grown configuration of approximately 4.65 million parameters and was trained on an NVIDIA Quadro M1200 with 4 GB of VRAM.

This result is an initial demonstration that the proposed architecture can be trained and can adapt its capacity during learning. It is not evidence that AFNN is already faster, smaller, more accurate, or more efficient than existing models. The central research question is whether a dynamically evolving computational organization can provide a better performance–resource trade-off than a fixed architecture under the same data, hardware, and training budget.

## 1. Motivation

In a conventional deep-learning workflow, the researcher chooses the architecture before training. The number of layers, channels, branches, and connections is fixed. Optimization then adjusts the weights, but the model cannot normally decide that it needs a new branch or that an existing branch has become unnecessary.

This separation between architecture design and parameter learning creates three practical limitations:

1. **Capacity must be estimated in advance.** If the model is too small, it may underfit. If it is too large, it may waste memory and computation or overfit the training data.
2. **The wiring is fixed during training.** The model cannot directly respond to a change in learning behavior by changing its computational organization.
3. **Branch-combination rules are usually predetermined.** A designer may choose addition, concatenation, or averaging, but the model does not generally learn how much each branch should contribute.

AFNN explores whether these decisions can become partly adaptive. It starts with a smaller structure, learns the relative contribution of its branches, monitors training progress, and can increase or reduce structural capacity under explicit constraints.

The purpose is not to claim that dynamic structure is automatically better. The purpose is to provide a concrete, reproducible system in which that hypothesis can be measured.

## 2. Research Question

> **Can a dynamically adapting fractal organization of neural computation achieve a competitive performance–resource trade-off compared with a matched fixed architecture?**

The comparison must be made under equivalent conditions. A meaningful evaluation should use the same dataset, input resolution, train–validation split, hardware, training budget, evaluation protocol, and, where possible, multiple random seeds.

Parameter count alone is not a sufficient baseline. Two models with the same number of parameters can have different activation memory, memory-access behavior, operation counts, training times, and inference latency. AFNN should therefore be evaluated as a complete computational system rather than only as a parameter total.

## 3. AFNN Architecture

### 3.1 Recursive fractal block

AFNN builds its main computational blocks recursively. A simplified description is:

```text
FractalBlock(depth d, branches b)
    = WeightedMerge(
          ConvBlock(x),
          FractalBlock(depth d-1)(x),
          ...                         # b - 1 recursive branches
      )

FractalBlock(depth 0) = ConvBlock
                        # two 3 × 3 convolutions with normalization and activation
```

Every branch receives the same input, but each branch can contain a different recursively constructed substructure. The repeated rule creates a self-similar organization: the same type of block appears at multiple levels of the network.

At each merge node, the branch outputs are combined using learned softmax weights:

```text
out(x) = sum_i softmax(logits)_i * branch_i(x)
```

This means that branch importance is learned from data. The merge is not a fixed mean and does not require the designer to decide in advance that every branch should contribute equally.

### 3.2 Network layout

The complete model follows this general flow:

```text
image
  -> stem convolution
  -> Stage 1: FractalBlock
  -> stride-2 convolution
  -> Stage 2: FractalBlock
  -> global average pooling
  -> linear classifier
  -> class scores
```

The recursive blocks create a hierarchical mixture of computational paths. Their merge weights also provide an inspectable signal: they show the relative contribution assigned to different branches during training and inference.

### 3.3 What makes AFNN different

AFNN is not intended to be a smaller version of a conventional CNN. Its defining feature is the organization of computation:

- A conventional CNN usually keeps its layer graph fixed while weights are optimized.
- A residual network adds fixed shortcut patterns selected by the designer.
- A fractal-style network uses recursive paths, but the architecture is normally selected before training.
- AFNN combines recursive organization with learned branch weighting and a controller that can modify branch capacity during training.

These differences define a research hypothesis. They do not, by themselves, prove better accuracy or lower cost.

## 4. Dynamic Structural Control

AFNN includes a control loop that observes learning behavior and manages the structure under explicit constraints. The intended training process is:

1. Start with a relatively compact fractal structure.
2. Train the model and monitor validation progress.
3. Detect a sustained lack of improvement using configured patience and minimum-delta values.
4. Add a branch if the parameter and branch budgets allow it.
5. Continue training while giving the new branch a controlled initialization and optimizer treatment.
6. Rejuvenate branches whose outputs have collapsed toward constant behavior.
7. Optionally prune branches with persistently low merge weights.

The main mechanisms are:

### Growth

When validation loss stops improving according to the configured criteria, AFNN can add one branch to the root block of each stage. Growth is limited by settings such as `max_branches_per_stage` and `max_total_params`. The new branch is initialized so that the network output changes only slightly at the growth event. This is intended to make growth function-preserving rather than disruptive.

### Rejuvenation

AFNN tracks branch-output statistics using an exponential moving average. If a branch becomes nearly constant, it can be reinitialized so that it has another opportunity to learn useful features.

### Pruning

The `prune_to_top_k` operation can remove branches with the smallest merge weights. The first two branches of a block are protected by the current implementation so that pruning does not remove the minimum structural path.

### Load-balancing regularization

A load-balancing loss discourages one branch from receiving almost all of the merge weight. This helps prevent inactive branches from being starved before they can learn.

### Top-k inference

The `hard_forward` operation can evaluate only the most heavily weighted branches. This provides a possible accuracy–compute trade-off at inference time, although the trade-off must be measured experimentally for each task.

## 5. Initial Experimental Result

The current repository reports one proof-of-concept training run with the following configuration:

| Item | Reported value |
|---|---|
| Dataset | 94 classes; approximately 56,000 images |
| Input resolution | 48 × 48 pixels |
| Validation split | 15% |
| Fractal depth | 3 |
| Initial branches | 3 per stage |
| Grown branches | 4 per stage |
| Parameters at grown configuration | Approximately 4.65 million |
| Validation accuracy | 74.1% |
| Clean training accuracy | Approximately 82% |
| Hardware | NVIDIA Quadro M1200 |
| GPU memory | 4 GB VRAM |
| Epoch time | Approximately 40–45 minutes |

The experiment shows that AFNN can train on a real multi-class image task, learn useful predictive behavior, and change its branch capacity during training. It does not show that AFNN is universally better than another architecture.

The reported result should therefore be interpreted as an **initial feasibility observation**. It establishes that the proposed design is implementable and trainable under limited hardware. It does not establish a state-of-the-art result or a general efficiency advantage.

## 6. What Must Be Compared

The meaningful baseline is not simply “which model has more parameters?” The important comparison is how different computational organizations behave under the same constraints.

A first controlled study should include:

1. A compact fixed CNN with standard training.
2. The same CNN with a tuned learning-rate schedule.
3. The same CNN with stronger regularization.
4. AFNN with dynamic growth enabled.
5. AFNN with dynamic growth disabled.

All models should use the same dataset, preprocessing, image size, train–validation split, hardware, number of training epochs or compute budget, and evaluation procedure. Multiple random seeds should be used when practical.

The study should report both predictive quality and resource behavior:

| Category | Measurements |
|---|---|
| Predictive quality | Accuracy, precision, recall, F1 score, and per-class results |
| Training cost | Wall-clock time, time per epoch, total compute, and convergence behavior |
| Inference cost | Latency, throughput, active branches, and MACs or FLOPs |
| Memory | Peak GPU VRAM, system RAM, and memory change during growth |
| Structural behavior | Growth events, pruning events, rejuvenation events, and final branch count |
| Robustness | Results across random seeds and sensitivity to configuration choices |

This evaluation will determine whether AFNN provides a useful trade-off, and in which operating conditions its dynamic structure helps or hurts.

## 7. Hypotheses

The project tests the following hypotheses:

- **H1 — Adaptive capacity:** adding branches when learning stalls can improve or recover validation performance.
- **H2 — Initial resource control:** starting with a smaller structure can avoid unnecessary early capacity.
- **H3 — Structural specialization:** different branches can develop different contributions to the task.
- **H4 — Controlled expansion:** explicit parameter and branch budgets can prevent unbounded structural growth.
- **H5 — Resource–performance trade-off:** AFNN may achieve a competitive result with a different balance of accuracy, training cost, memory, and inference cost than a fixed architecture.

These are testable hypotheses, not established results.

## 8. Limitations and Scientific Scope

The current evidence has several limitations. It is based on one reported dataset, one main training run, one hardware configuration, and no matched baseline study. The dataset description, class distribution, and complete reproducibility settings should be expanded in a future report.

The current result does not justify claims that AFNN is faster, more memory efficient, more scalable, or more accurate than CNNs, ResNets, or FractalNet. It also does not establish performance on large datasets such as ImageNet.

The main current contribution is a working implementation of a different architectural idea: learned branch combination combined with structural growth, pruning, and rejuvenation. The comparative contribution remains to be established through controlled experiments.

## 9. Reproducibility Plan

A complete follow-up report should record the exact dataset name and version, source, class distribution, file filtering rules, preprocessing, augmentation, optimizer, learning-rate schedule, batch size, number of epochs, random seeds, software versions, hardware, AFNN configuration, growth criteria, pruning criteria, rejuvenation criteria, and stopping rule.

The repository already contains training scripts, inference utilities, a GUI, tests, configuration documentation, and dataset-packing tools. These components are intended to make the next comparison study easier to reproduce.

## 10. Future Work

The next development and research steps are:

1. Document the current dataset and training run in full.
2. Implement and train a matched compact CNN baseline.
3. Add fixed-architecture controls for scheduling and regularization.
4. Run AFNN ablations with growth, pruning, rejuvenation, and load balancing disabled separately.
5. Measure accuracy, training time, peak VRAM, MACs, inference latency, and structural changes.
6. Repeat the experiments with multiple random seeds.
7. Analyze whether the learned branch weights correspond to useful specialization.
8. Evaluate larger datasets only after the controlled baseline study is complete.
9. Publish the final measurements and code used to generate the comparison tables.

## 11. Conclusion

AFNN investigates whether a neural network can learn not only its parameter values but also an appropriate amount and organization of its computational structure while learning the task itself.

Its proposed mechanism combines a recursively organized fractal architecture, learned softmax branch merging, and a control loop for growth, pruning, and branch rejuvenation. This makes AFNN a concrete platform for studying dynamic neural architecture rather than a claim that fixed architectures are already obsolete.

The current 74.1% validation result demonstrates feasibility under limited hardware. It does not establish superiority over existing models. The decisive next step is a matched comparison in which AFNN and fixed baselines use the same data, hardware, training budget, and evaluation protocol.

> **Research question:** Can a neural network learn not only its weights, but also an effective amount and organization of computational structure while it learns the task?

## Project Resources

- **GitHub repository:** <https://github.com/ahmedhjkj/AFNN>
- **Hugging Face repository:** <https://huggingface.co/Ahmedethfw/AFNN>
- **Configuration guide:** [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md)
- **API reference:** [`docs/API.md`](docs/API.md)
- **License:** MIT License

## References

[1]: https://arxiv.org/abs/1605.07648 "FractalNet: Ultra-Deep Neural Networks without Residuals"

[2]: https://arxiv.org/abs/1512.03385 "Deep Residual Learning for Image Recognition"

[3]: https://pytorch.org/docs/stable/ "PyTorch Documentation"

**Author:** Ahmed Al-Amin  
**Project:** AFNN — Adaptive Fractal Neural Network

---

*This document is a technical whitepaper draft. Comparative claims should be updated when controlled baseline experiments are completed.*
