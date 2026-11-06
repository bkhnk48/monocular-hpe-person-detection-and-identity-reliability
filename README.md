# Monocular HPE: Person Detection and Identity Reliability

**Research on false person detections, missed persons, and identity inconsistency in monocular 3D human pose estimation.**

---

## 1. Overview

Monocular 3D Human Pose Estimation (HPE) systems aim to reconstruct 3D human poses from video captured by a single camera. Many modern systems also produce person-level predictions and maintain person identities across frames.

However, their outputs may not consistently correspond to the actual people present in the scene.

A model may:

- Predict a person when no corresponding ground-truth (GT) person exists.
- Fail to predict a person who is actually present in the scene.
- Assign detections belonging to different real people to the same predicted identity.
- Split detections belonging to one real person across multiple predicted identities.
- Produce an inconsistent number of person hypotheses across consecutive frames.

These problems are not merely pose estimation errors. They concern the reliability of the model's person hypotheses and its temporal identity associations.

This project investigates these phenomena systematically, using frame-level person detections, 3D pose information, camera calibration, and ground-truth annotations.

The initial experimental setting is **WHAM evaluated on Harmony4D**, with the intention of developing a methodology that can be extended to other monocular HPE systems and datasets.

### 1.1. Research Problem

Let:

- \(t \in \{1,\ldots,T\}\) denote a video frame.
- \(G_t\) denote the set of ground-truth people in frame \(t\).
- \(D_t\) denote the set of person detections produced by an HPE model in frame \(t\).

The model's detections do not necessarily have a one-to-one correspondence with the GT people.

In particular:

\[
|D_t| \neq |G_t|
\]

may hold for some or even many frames.

Furthermore, a model-provided identity label does not necessarily represent a consistent real-world identity throughout the sequence.

The central research problem is therefore:

> How can we characterize, quantify, and analyze inconsistencies between predicted person hypotheses and ground-truth human identities in monocular 3D HPE systems?

---

## 2. Research Objectives

This project has four main objectives:

1. **Characterize person detection errors:** Identify false-positive detections and missed ground-truth people.
2. **Analyze identity inconsistency:** Determine whether predicted identities consistently correspond to real people across time.
3. **Develop a global temporal association method:** Formulate a multi-commodity optimization approach that jointly associates detections with GT identities while allowing missing and unmatched detections.
4. **Evaluate model reliability:** Compare the original model output with several frame-level and sequence-level association baselines.

The project distinguishes two related but different questions:

- How accurately does a model estimate a human pose?
- How reliably does a model identify and maintain the identity of the person represented by that pose?

A model may perform well on pose estimation while still producing unreliable person-level associations.

---

## 3. Research Questions

### RQ1. Person Detection Reliability

How frequently do monocular 3D HPE systems produce:

- False-positive person detections?
- False-negative or missed-person events?
- Incorrect numbers of person hypotheses per frame?

This question focuses on the discrepancy between predicted person detections and the actual people present in the scene.

### RQ2. Temporal Identity Consistency

To what extent do model-provided person identities correspond to consistent ground-truth identities over time?

In particular:

- Does one predicted identity contain detections from multiple GT people?
- Is one GT person represented by multiple predicted identities?
- How frequently do identity changes or identity fragmentation occur?

### RQ3. Global Temporal Association

Can a global temporal optimization method, specifically Multi-Commodity Min-Cost Flow (MC-MCF), produce more consistent associations between frame-level detections and GT identities than independent frame-level assignment methods?

The optimization should explicitly allow:

- A GT person to have no detection in a frame.
- A predicted detection to correspond to no GT person.
- The number of predicted detections to differ from the number of GT people.

### RQ4. Relationship Between Pose Quality and Identity Reliability

Are person detection and identity association errors correlated with 3D pose quality?

For example:

- Do detections with larger pose errors also exhibit more identity inconsistency?
- Can geometrically plausible poses still be associated with the wrong person?
- Are some identity errors caused by occlusion, similar body configurations, or temporary detection failures?

---

## 4. Problem Formulation

### 4.1. Frame-Level Detection Representation

Each predicted person pose is treated as an independent frame-level detection.

Define:

\[
d_{t,i} = (t, i, P_{t,i})
\]

where:

- \(t\) is the frame index.
- \(i\) is the detection identifier within the frame.
- \(P_{t,i}\) is the predicted 3D skeleton.

For WHAM, the original `body_id` is retained as metadata, but it is **not assumed to be a ground-truth identity**.

In particular:

\[
body\_id \not\equiv PersonID
\]

A WHAM body ID may contain detections corresponding to different GT people over time.

Similarly, one GT person may be represented by multiple WHAM body IDs.

Therefore, the proposed association method operates on frame-level detections rather than assuming that the original WHAM IDs are correct tracks.

### 4.2. Ground-Truth Representation

Let:

\[
g_{t,j} = (t,j,Q_{t,j})
\]

where:

- \(j\) is a GT person identity.
- \(Q_{t,j}\) is the GT 3D skeleton.

A GT person may be absent from the available predictions in a particular frame.

### 4.3. Assignment Variables

For each GT person \(j\), frame \(t\), and detection \(i\), define:

\[
x_{t,j,i} \in \{0,1\}
\]

where \(x_{t,j,i}=1\) means detection \(i\) is assigned to GT person \(j\).

Define a missed-detection variable:

\[
m_{t,j} \in \{0,1\}
\]

where \(m_{t,j}=1\) means GT person \(j\) has no assigned detection in frame \(t\).

Define an unmatched-detection variable:

\[
u_{t,i} \in \{0,1\}
\]

where \(u_{t,i}=1\) means detection \(i\) is not assigned to any GT person.

### 4.4. Assignment Constraints

Each GT person must have exactly one state in each frame: either one assigned detection or a missed-detection state.

\[
\sum_{i \in D_t} x_{t,j,i} + m_{t,j} = 1
\]

Each predicted detection can be assigned to at most one GT person.

\[
\sum_{j \in G_t} x_{t,j,i} + u_{t,i} = 1
\]

These constraints explicitly support:

- Missing predictions for real people.
- Extra predictions without a corresponding GT person.
- Different numbers of detections and GT people.
- One-to-one assignments between detections and GT identities within each frame.

The model does not force every detection to match a GT person.

---

## 5. Methodology: Multi-Commodity Min-Cost Flow

### 5.1. Why Multi-Commodity Optimization?

Independent frame-level matching cannot fully exploit temporal information.

For example, two people may have similar poses in a particular frame. A frame-level assignment may associate their detections incorrectly even though their positions and movements over neighboring frames provide useful identity evidence.

A global temporal optimization can consider the entire sequence when deciding which detections correspond to each GT identity.

We formulate the problem as a coupled multi-commodity optimization problem.

Each GT identity is treated as a separate commodity.

The commodities share the same detection resources, so a detection cannot be assigned to multiple GT people.

This shared-capacity constraint couples the individual identity-assignment problems.

### 5.2. Network Structure

The network is constructed using frame-level detections.

For every GT identity \(j\), the network contains possible states at each frame:

- Detection state: assign a specific detection to GT person \(j\).
- Missed state: assign no detection to GT person \(j\).

Temporal edges connect states belonging to consecutive frames.

The conceptual structure is:

```text
Frame t                  Frame t+1

Detection 1  ----------> Detection 1
     |                         |
     +----------------------> Detection 2

Detection 2  ----------> Detection 1
     |                         |
     +----------------------> Detection 2

Missed state  ----------> Detection states
     |
     +--------------------> Missed state
```

This structure is instantiated for every GT identity.

All identity-specific networks share detection-capacity constraints.

The final formulation is a coupled integer optimization model. It should not be treated as an ordinary independent single-commodity flow problem.

### 5.3. Objective Function

The proposed objective combines unary assignment costs, temporal transition costs, and unmatched-detection penalties:

\[
\begin{aligned}
\min \mathcal{L} =\;&
\sum_{t,j} C^U_j(a_{t,j}) \\
&+ \lambda_T
\sum_{t=1}^{T-1}
\sum_j C^T_j(a_{t,j},a_{t+1,j})\\
&+ c_{FP}\sum_{t,i}u_{t,i}
\end{aligned}
\]

where:

- \(a_{t,j}\) represents the state selected for GT person \(j\) in frame \(t\).
- \(C^U_j\) is the unary assignment cost.
- \(C^T_j\) is the temporal transition cost.
- \(c_{FP}\) is the penalty for an unmatched detection.
- \(\lambda_T\) controls the importance of temporal consistency.

The unary cost includes either a detection-to-GT matching cost or a missed-detection penalty.

### 5.4. Unary Matching Cost

For a detection assigned to a GT person, the matching cost may combine:

\[
C^{match}_{t,j,i}
=
\lambda_R C^{root}_{t,j,i}
+
\lambda_P C^{pose}_{t,j,i}
+
\lambda_{2D} C^{reproj}_{t,j,i}
\]

Possible components include:

**Root-position cost**

Measures the difference between predicted and GT root positions:

\[
C^{root}_{t,j,i}
=
\|r(P_{t,i})-r(Q_{t,j})\|_2^2
\]

**Root-relative pose cost**

Measures the difference between corresponding joints after subtracting the root position:

\[
C^{pose}_{t,j,i}
=
\sum_k w_k
\|
(P_{t,i,k}-r(P_{t,i}))
-
(Q_{t,j,k}-r(Q_{t,j}))
\|_2^2
\]

**Reprojection cost**

Projects the predicted 3D skeleton into the camera coordinate system and compares it with corresponding 2D observations, when reliable 2D observations are available.

The actual implementation must explicitly handle:

- Joint correspondence between WHAM and Harmony4D.
- Coordinate-system conventions.
- Camera extrinsics and intrinsics.
- Units and scale differences.
- Missing or invalid joints.

Per-detection Procrustes alignment should not automatically be used for association, because it can remove position and orientation differences that are important for distinguishing people.

For missed detections:

\[
C^U_j(\bot)=c_{FN}
\]

where \(c_{FN}\) is the missed-person penalty.

### 5.5. Temporal Transition Cost

Temporal transition costs encourage consistent assignments across adjacent frames.

Possible transitions include:

| Previous state | Current state | Interpretation |
|---|---|---|
| Detection | Detection | Continue tracking the same GT person |
| Detection | Missed | Temporary missed detection |
| Missed | Detection | Reacquisition |
| Missed | Missed | Continued absence of a prediction |

For detection-to-detection transitions, the cost can include:

- Root displacement consistency.
- Root velocity consistency.
- Root-relative pose changes.
- Motion consistency.
- Camera-aware geometric constraints.

The missed-to-missed transition may have zero additional cost if the missed-detection penalty is already included in the unary term. This avoids double-counting missing-person penalties.

### 5.6. Camera Information

Camera calibration is available in the initial WHAM–Harmony4D experiments.

Camera information can be used to:

- Transform predictions and GT poses into a common coordinate system.
- Compute camera-aware geometric distances.
- Reproject 3D predictions into image coordinates.
- Evaluate whether an association is geometrically plausible.

All coordinates must be converted into compatible coordinate systems and units before computing matching costs.

### 5.7. Optimization and Solver

The initial implementation will formulate the assignment problem as a binary integer optimization model.

Candidate solver approaches include:

- Mixed-Integer Linear Programming (MILP).
- Constraint Programming / CP-SAT.
- A specialized multi-commodity flow implementation, if its constraints and objective can represent the shared detection capacities correctly.

The selected solver must return globally feasible assignments satisfying all detection and GT identity constraints.

---

## 6. Experimental Baselines

We define six baseline configurations.

### B1. Greedy Frame-Level Matching

Each GT person is matched to the lowest-cost available detection in an individual frame.

A conflict-resolution mechanism is required to prevent multiple GT identities from selecting the same detection.

Characteristics:

- Independent frame-level decisions.
- No sequence-level optimization.
- Simple and computationally inexpensive.
- Vulnerable to local assignment errors and identity switching.

Purpose:

Establish a simple reference point for evaluating more structured assignment methods.

### B2. Hungarian Frame-Level Assignment

Use the Hungarian algorithm to solve a one-to-one assignment problem independently for each frame.

The cost matrix contains GT-to-detection matching costs.

Dummy assignments are introduced to represent:

- GT people without detections.
- Detections without GT matches.

The resulting assignment minimizes the total frame-level matching cost.

Characteristics:

- Globally optimal within each frame.
- Enforces one-to-one assignment within a frame.
- Does not explicitly enforce temporal identity consistency.

Purpose:

Measure the benefit of optimal frame-level assignment over greedy matching.

### B3. Single-Commodity Min-Cost Flow

Use a sequence-level temporal optimization model based on a single-commodity flow formulation.

Unlike B2, this baseline considers temporal connections between detections.

The formulation must explicitly define how detection paths, missed detections, and shared detection capacities are represented.

Characteristics:

- Uses temporal information.
- Performs sequence-level optimization.
- Does not explicitly model a separate commodity for every GT identity in the same way as B4.

Purpose:

Evaluate whether global temporal optimization alone improves association quality, and whether identity-specific multi-commodity constraints provide additional benefits.

**Important implementation note:** B3 must be formulated as a genuinely distinct single-commodity model. It must not simply be another name for the coupled multi-commodity formulation.

### B4. Proposed Multi-Commodity Min-Cost Flow

This is the proposed method.

Each GT identity is represented as an individual commodity.

Each commodity selects one state per frame:

- A detection assigned to that identity.
- A missed-detection state.

All commodities share detection-capacity constraints.

Consequently, a detection cannot be assigned to multiple GT identities.

The optimization jointly considers:

- Unary detection-to-GT matching costs.
- Temporal transition costs.
- False-positive penalties.
- Missed-detection penalties.

Characteristics:

- Global sequence-level optimization.
- Explicit GT identity modeling.
- Explicit handling of missing and unmatched detections.
- Joint assignment across all GT identities.

Purpose:

Evaluate whether jointly optimizing identity-specific assignments improves detection attribution and temporal identity consistency.

### B5. Oracle Frame-Level Assignment

This baseline provides an oracle diagnostic reference for frame-level assignment.

It uses the same one-to-one assignment structure as B2 but is allowed to use idealized GT-informed information when constructing or resolving matching costs.

For example, an oracle matching cost may use information that would not be available to a practical prediction-only association system.

Characteristics:

- Frame-level assignment.
- No temporal optimization.
- GT-informed or idealized matching information.
- Represents an oracle diagnostic reference rather than a deployable method.

Purpose:

Estimate how much error remains when temporal information is not considered but frame-level matching information is made more favorable.

**Important distinction:** If B2 and B5 use exactly the same cost matrix, dummy penalties, and assignment constraints, they are algorithmically identical. In that case, they must be merged rather than reported as separate baselines.

B5 should only be retained as a separate baseline when its oracle information is explicitly defined and documented.

It is not automatically an upper bound on every possible sequence-level method.

### B6. Original WHAM Identity Evaluation

Evaluate the original WHAM predictions using their unchanged `body_id` assignments.

No identity correction or temporal re-association is performed.

The original WHAM IDs are mapped to GT identities only for metric computation, using a global one-to-one identity mapping with dummy assignments.

This mapping is an evaluation procedure, not a modification of WHAM predictions.

The evaluation should measure:

- Identity true positives.
- Identity false positives.
- Identity false negatives.
- IDF1.
- Identity purity.
- Identity fragmentation.

For a predicted identity \(h\), define its identity purity as:

\[
Purity(h)=
\frac{\max_j N_{h,j}}
{\sum_j N_{h,j}}
\]

where \(N_{h,j}\) is the number of matched detections assigned to predicted identity \(h\) that correspond to GT identity \(j\).

Identity contamination occurs when one predicted identity contains detections from multiple GT people.

Identity fragmentation occurs when one GT person is represented by multiple predicted identities.

These are different error types and should be reported separately.

Purpose:

Establish the identity reliability of the original WHAM output before applying any proposed association method.

---

## 7. Evaluation Metrics

Detection, identity association, and pose accuracy should be evaluated separately.

### 7.1. Person Detection Metrics

Recommended metrics:

- Detection Precision.
- Detection Recall.
- Detection F1-score.
- False-positive count.
- False-negative count.
- False-positive rate.
- Missed-person rate.

A predicted detection is considered a true positive only if it satisfies the predefined geometric matching criterion with a GT person.

The matching threshold and coordinate system must be specified.

### 7.2. Identity Metrics

Recommended metrics:

- IDTP.
- IDFP.
- IDFN.
- IDF1.
- Identity purity.
- Identity contamination.
- Identity fragmentation.
- Identity switches, where applicable.

For example:

\[
IDF1 =
\frac{2IDTP}
{2IDTP+IDFP+IDFN}
\]

The identity mapping used for B6 must be computed consistently and documented.

### 7.3. Pose Accuracy Metrics

Recommended metrics:

- MPJPE.
- PA-MPJPE.
- Root-position error.
- Per-joint error.
- Temporal acceleration error, where applicable.

Pose accuracy should be computed on valid matched person-frame pairs.

The matching procedure must be reported because different association methods may produce different sets of matched pairs.

### 7.4. Temporal Consistency Metrics

Additional metrics may include:

- Number of identity switches.
- Number of identity fragmentation events.
- Average track duration.
- Detection continuity.
- Reacquisition success after missed detections.

These metrics help characterize errors that cannot be fully described by per-frame pose accuracy.

---

## 8. Experimental Design

### 8.1. Initial Dataset

The initial experiments use:

- Dataset: Harmony4D.
- Model: WHAM.
- Input: monocular video and camera calibration.
- Predictions: WHAM 3D skeleton outputs.
- Ground truth: Harmony4D 3D human pose annotations.

The experimental pipeline should preserve:

- Video and sequence identifiers.
- Camera identifiers.
- Frame indices.
- Original WHAM `body_id`.
- Unique frame-level detection identifiers.
- GT `PersonID`.
- Joint coordinates.
- Camera calibration metadata.

### 8.2. Data Separation

Experiments should be separated by subject or sequence rather than randomly splitting individual frames.

This helps prevent temporal leakage between training, validation, and test sets.

If optimization weights and penalties are selected using validation data, the final test set must remain untouched during parameter tuning.

### 8.3. Ablation Studies

Potential ablation studies include:

1. Root-position cost only.
2. Root-relative pose cost only.
3. Root-position plus pose cost.
4. Addition of camera-aware reprojection cost.
5. Without temporal transition cost.
6. With different missed-detection penalties.
7. With different unmatched-detection penalties.
8. Different temporal transition models.

These experiments can identify which components contribute most to association reliability.

### 8.4. Robustness Analysis

The method should also be evaluated under different conditions:

- Different numbers of GT people.
- Different numbers of predicted detections.
- Temporary occlusions.
- Long missed-detection intervals.
- Similar body poses between people.
- Identity fragmentation in original model outputs.
- Different camera viewpoints.

---

## 9. Expected Research Contributions

The project aims to investigate and establish the following contributions:

1. **A systematic characterization of person-level reliability errors in monocular 3D HPE**, distinguishing false-positive detections, missed people, identity contamination, and identity fragmentation.

2. **A frame-level detection representation** that does not assume model-provided person IDs are temporally correct.

3. **A GT-assisted global temporal association formulation** that explicitly allows missing GT detections and unmatched predicted detections.

4. **A comparative evaluation framework** covering greedy matching, Hungarian assignment, single-commodity temporal optimization, multi-commodity optimization, oracle frame-level assignment, and original model identity evaluation.

5. **An analysis of the relationship between pose accuracy and person identity reliability**, investigating whether good 3D pose estimates necessarily imply reliable person-level predictions.

These are research objectives and hypotheses, not established experimental findings. Their validity must be demonstrated through experiments.

---

## 10. Project Structure

```text
monocular-hpe-person-detection-and-identity-reliability/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── calibration/
│
├── src/
│   ├── data/
│   │   ├── wham_loader.py
│   │   ├── harmony4d_loader.py
│   │   └── coordinate_transform.py
│   │
│   ├── costs/
│   │   ├── root_cost.py
│   │   ├── pose_cost.py
│   │   ├── reprojection_cost.py
│   │   ├── transition_cost.py
│   │   └── cost_matrix.py
│   │
│   ├── optimization/
│   │   ├── network_builder.py
│   │   ├── mc_mcf_model.py
│   │   ├── constraints.py
│   │   └── solver.py
│   │
│   ├── evaluation/
│   │   ├── detection_metrics.py
│   │   ├── identity_metrics.py
│   │   ├── identity_purity.py
│   │   ├── contamination.py
│   │   └── pose_metrics.py
│   │
│   └── main.py
│
├── experiments/
│   ├── configs/
│   └── run_experiment.py
│
├── outputs/
│   ├── assignments/
│   ├── metrics/
│   └── visualizations/
│
├── tests/
│
├── docs/
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 11. Implementation Principles

The implementation should follow these principles:

1. **Never assume WHAM `body_id` equals GT `PersonID`.**
2. Treat every `(frame_id, body_id)` pair as a distinct frame-level detection.
3. Preserve original model IDs as metadata for evaluating the unmodified model output.
4. Allow GT people to have no corresponding detection in a frame.
5. Allow predicted detections to remain unmatched.
6. Enforce one-to-one detection assignment within each frame.
7. Convert all 3D coordinates into a common coordinate system before calculating geometric costs.
8. Separate association evaluation from pose accuracy evaluation.
9. Avoid using test-set information to tune optimization parameters.
10. Record all experiment configurations and solver settings for reproducibility.

---

## 12. Current Scope and Future Directions

### Current Scope

The initial scope is GT-assisted analysis of WHAM predictions on Harmony4D.

The proposed MC-MCF formulation uses GT identities and GT pose information to evaluate and optimize the attribution of predicted detections.

Therefore, the current method is an **evaluation and attribution framework**, not yet a fully automatic identity correction method that operates without GT.

### Future Directions

Potential extensions include:

- Applying the framework to additional monocular 3D HPE systems.
- Evaluating additional datasets and camera configurations.
- Developing GT-free association methods using only predicted 2D/3D poses and observable camera information.
- Investigating uncertainty-aware association costs.
- Studying the effect of occlusion and viewpoint changes.
- Developing methods to detect unreliable person hypotheses before they affect downstream applications.

---

## 13. Reproducibility and Data Policy

This repository contains research code, experiment configurations, evaluation tools, and documentation.

Large datasets, original videos, GT annotations, and model checkpoints are not included by default.

Users must obtain the corresponding datasets and model weights from their original sources and comply with their licenses and usage conditions.

All reported experiments should document:

- Dataset and sequence identifiers.
- Model version and configuration.
- Camera calibration.
- Coordinate conventions and units.
- Joint mapping.
- Association cost definitions.
- Optimization parameters.
- Solver and solver version.
- Evaluation thresholds.
- Random seeds, where applicable.

---

## 14. Research Status

**Status:** Research methodology and experimental pipeline under development.

**Initial case study:** WHAM on Harmony4D.

**Primary methodological direction:** Global temporal association using a coupled Multi-Commodity Min-Cost Flow formulation.

Experimental conclusions will be added after the corresponding baselines and evaluation pipeline have been implemented and validated.
