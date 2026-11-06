# Monocular HPE: Person Detection and Identity Reliability

Research project investigating person detection errors and identity inconsistency in monocular 3D Human Pose Estimation (HPE) models, with a focus on WHAM and the development of a Multi-Commodity Min-Cost Flow (MC-MCF) framework.

## 1. Research Motivation

Monocular 3D Human Pose Estimation models estimate 3D human poses from video sequences. However, their outputs may contain errors that are not adequately described by conventional pose estimation metrics such as MPJPE or PA-MPJPE.

In multi-person scenarios, a model may:

- Detect a person who does not correspond to any Ground Truth (GT) person.
- Fail to detect a person who is present in the Ground Truth.
- Assign detections from different real people to the same predicted identity.
- Split detections belonging to one real person across multiple predicted identities.
- Produce inconsistent person detections across consecutive frames.

These errors affect both person-level detection reliability and temporal identity consistency.

This project investigates these phenomena systematically and explores optimization-based approaches to analyze and improve the association between predicted human poses and actual people.

## 2. Research Questions

The project focuses on the following research questions:

- **RQ1 — Detection Reliability:** How frequently do monocular HPE models produce false-positive and false-negative person detections?
- **RQ2 — Identity Consistency:** To what extent do predicted person identities remain consistent with actual people across video frames?
- **RQ3 — Temporal Association:** Can temporal and geometric information improve the association between frame-level pose detections and Ground Truth identities?
- **RQ4 — Optimization Framework:** Can a Multi-Commodity Min-Cost Flow formulation provide a more reliable global assignment than frame-independent matching methods?

## 3. Research Dataset and Models

### MPI-INF-3DHP

The initial experimental dataset is MPI-INF-3DHP.

It provides 3D human pose annotations and camera information that can be used to evaluate monocular HPE predictions.

The initial experiments will use:

- WHAM predictions generated from MPI-INF-3DHP videos.
- MPI-INF-3DHP Ground Truth annotations.
- Camera calibration information.
- Frame-level and person-level identifiers.

The initial implementation will focus on matching the actual CSV formats produced for WHAM predictions and MPI-INF-3DHP Ground Truth.

### WHAM

WHAM is used as the initial monocular HPE model under investigation.

The original WHAM body IDs are retained as model outputs. They must not be assumed to represent Ground Truth person identities.

Each detection is represented by a frame-level observation:

\[
d_{t,i}
\]

where \(t\) denotes the frame and \(i\) denotes a detection within that frame.

A WHAM body ID is retained as metadata associated with the detection.

### Future Extension: Harmony4D

Harmony4D is planned as an additional evaluation dataset.

The data-loading and coordinate-processing architecture is designed to support multiple datasets, but the first implementation and parameter optimization will focus on MPI-INF-3DHP.

## 4. Problem Formulation

Let:

- \(G_t\) be the set of Ground Truth people in frame \(t\).
- \(D_t\) be the set of model detections in frame \(t\).
- \(g_{t,j}\) be Ground Truth person \(j\).
- \(d_{t,i}\) be predicted detection \(i\).

The association problem is to determine whether each predicted detection corresponds to a Ground Truth person.

The formulation explicitly allows:

1. A Ground Truth person to have no corresponding detection in a frame.
2. A predicted detection to have no corresponding Ground Truth person.
3. A Ground Truth person to be associated with different model IDs across frames.
4. A model ID to contain detections corresponding to different Ground Truth people.

This avoids assuming that the model's original identity assignments are always correct.

## 5. Proposed MC-MCF Framework

The project investigates a Multi-Commodity Min-Cost Flow (MC-MCF) formulation for global temporal association.

Each Ground Truth identity is treated as a commodity.

The optimization operates on frame-level detections rather than assuming that the original WHAM tracks are correct.

### 5.1 Decision Variables

Let:

\[
x_{t,j,i}\in\{0,1\}
\]

indicate whether detection \(i\) in frame \(t\) is assigned to Ground Truth person \(j\).

Let:

\[
m_{t,j}\in\{0,1\}
\]

indicate whether Ground Truth person \(j\) is missed in frame \(t\).

Let:

\[
u_{t,i}\in\{0,1\}
\]

indicate whether detection \(i\) is unmatched.

Each Ground Truth person must either be assigned one detection or be marked as missed:

\[
\sum_{i\in D_t}x_{t,j,i}+m_{t,j}=1
\]

Each detection must either be assigned to one Ground Truth person or remain unmatched:

\[
\sum_{j\in G_t}x_{t,j,i}+u_{t,i}=1
\]

These constraints explicitly represent missed detections and unmatched predictions.

### 5.2 Unary Matching Cost

The unary cost combines three geometric components:

\[
C^U_{tji}
=
\alpha C^{pose}_{tji}
+
\beta C^{root}_{tji}
+
\gamma C^{2D}_{tji}
\]

where:

- \(C^{pose}\): root-relative 3D pose or skeleton-shape discrepancy.
- \(C^{root}\): 3D root-position discrepancy.
- \(C^{2D}\): 2D reprojection discrepancy.
- \(\alpha,\beta,\gamma\): non-negative weighting parameters.

The cost components should be normalized or otherwise made comparable before combining them.

An initial parameter search will investigate:

\[
\alpha+\beta+\gamma=1
\]

The parameters will be selected using validation data, not test data.

The feasibility of the 2D reprojection term depends on the availability of suitable 2D observations and camera calibration.

### 5.3 Temporal Cost

Temporal costs measure the consistency of assignments across consecutive frames.

A general objective is:

\[
\min \mathcal{L}
=
\sum_{t,j}C^U_{t,j}
+
\lambda_T\sum_{t,j}C^T_{t,j}
+
c_{FP}\sum_{t,i}u_{t,i}
\]

where:

- \(C^U\): unary matching and missed-detection costs.
- \(C^T\): temporal transition costs.
- \(\lambda_T\): temporal-cost weight.
- \(c_{FP}\): unmatched-detection penalty.

The missed-detection penalty is included in the unary cost when a Ground Truth person is assigned to the missed state.

The exact network construction and optimization constraints will be documented as the implementation develops.

## 6. Experimental Baselines

The proposed MC-MCF method should be compared with simpler association strategies.

| Method | Description |
|---|---|
| B1: Greedy Matching | Frame-level matching using greedy assignment. |
| B2: Hungarian Matching | Frame-level global assignment with dummy assignments for missed and unmatched observations. |
| B3: Temporal Optimization | Sequence-level association using temporal information. |
| B4: MC-MCF | Proposed multi-commodity optimization with shared detection-capacity constraints. |
| B5: Oracle Assignment | An explicitly defined GT-informed reference or upper-bound experiment, if it provides information beyond B2. |
| B6: Original WHAM IDs | Evaluation of the model's original identity assignments without modifying them. |

Baselines must be implemented with clearly documented costs, constraints, and information available to each method.

If two baselines use the same cost matrix and assignment constraints, they should be treated as the same method rather than presented as separate contributions.

## 7. Evaluation Metrics

### 7.1 Detection Metrics

- Precision
- Recall
- F1-score
- False-positive detections
- Missed detections

### 7.2 Identity Metrics

- IDTP
- IDFP
- IDFN
- IDF1
- Identity purity
- Identity contamination
- Identity fragmentation

Original WHAM IDs will be evaluated separately from the optimized GT-to-detection assignments.

### 7.3 Pose Metrics

- MPJPE
- PA-MPJPE
- Root-position error
- Acceleration error

Detection and identity metrics evaluate whether the model identifies and associates people correctly.

Pose metrics evaluate the geometric accuracy of the predicted skeletons.

These metrics should be reported separately because good pose accuracy does not necessarily imply reliable person detection or identity consistency.

## 8. Hyperparameter Optimization

The first optimization experiment will focus on finding suitable values for:

\[
(\alpha,\beta,\gamma)
\]

using MPI-INF-3DHP validation data.

The procedure is:

1. Load WHAM predictions from CSV files.
2. Load MPI-INF-3DHP Ground Truth from CSV files.
3. Map corresponding joints between the two representations.
4. Validate coordinate systems, coordinate units, camera information, and frame indexing.
5. Compute the individual pose, root-position, and reprojection costs.
6. Normalize the cost components.
7. Search candidate values of \(\alpha,\beta,\gamma\).
8. Run the association method for each candidate.
9. Evaluate the resulting assignments using validation metrics.
10. Select parameters according to a predefined validation objective.

Other parameters, including temporal weights and false-positive/false-negative penalties, should initially be held fixed when investigating the effect of \(\alpha,\beta,\gamma\).

The parameter-selection objective should be based on an evaluation metric such as IDF1 or a predefined association score, rather than simply selecting the smallest optimization cost.

Data splitting should be performed at the subject or sequence level where possible, rather than randomly splitting individual frames.

## 9. Project Structure

```text
monocular-hpe-person-detection-and-identity-reliability/
│
├── data/
│   ├── raw/
│   │   └── mpi_inf_3dhp/
│   │       ├── wham_predictions/
│   │       └── ground_truth/
│   │
│   ├── processed/
│   │   └── mpi_inf_3dhp/
│   │
│   └── calibration/
│       └── mpi_inf_3dhp/
│
├── src/
│   ├── data/
│   │   ├── wham_loader.py
│   │   ├── mpi_gt_loader.py
│   │   ├── mpi_joint_mapping.py
│   │   ├── harmony4d_loader.py
│   │   └── coordinate_transform.py
│   │
│   ├── costs/
│   │   ├── root_cost.py
│   │   ├── pose_cost.py
│   │   ├── reprojection_cost.py
│   │   ├── transition_cost.py
│   │   ├── unary_cost.py
│   │   └── cost_matrix.py
│   │
│   ├── optimization/
│   │   ├── network_builder.py
│   │   ├── mc_mcf_model.py
│   │   ├── constraints.py
│   │   └── solver.py
│   │
│   ├── tuning/
│   │   ├── validation.py
│   │   ├── parameter_search.py
│   │   └── objective.py
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
│   │   └── mpi_validation.yaml
│   ├── tune_mpi.py
│   └── run_experiment.py
│
├── outputs/
│   ├── parameter_search/
│   ├── assignments/
│   ├── metrics/
│   └── visualizations/
│
├── tests/
├── docs/
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── README.md
```

## 10. Data Organization

Raw dataset files should be stored separately from processed data.

Example:

```text
data/raw/mpi_inf_3dhp/
│
├── wham_predictions/
│   ├── RA-WHAM_thongTinVideo_cam1.csv
│   ├── RA-WHAM_thongTinVideo_cam2.csv
│   └── ...
│
└── ground_truth/
    ├── thongTinVideo_cam1.csv
    ├── thongTinVideo_cam2.csv
    └── ...
```

Processed data, intermediate assignments, and experiment outputs should not overwrite the original CSV files.

Large datasets and generated outputs should generally not be committed to Git. Their locations and download or generation instructions should be documented separately.

## 11. Reproducibility

Each experiment should record:

- Dataset and sequence identifiers.
- Camera identifiers.
- Model and inference configuration.
- Joint mapping configuration.
- Coordinate system and unit conventions.
- Cost normalization method.
- Values of \(\alpha,\beta,\gamma\).
- Temporal and unmatched-detection penalties.
- Solver and solver configuration.
- Validation and evaluation metrics.

Parameter-search results should be exported to CSV for further analysis.

Example:

```text
outputs/
└── parameter_search/
    └── mpi_validation_results.csv
```

## 12. Current Development Priorities

The implementation will proceed incrementally.

| Stage | Task | Status |
|---|---|---|
| 1 | Organize MPI-INF-3DHP WHAM and GT data | Planned |
| 2 | Implement WHAM CSV loader | Planned |
| 3 | Implement MPI Ground Truth CSV loader | Planned |
| 4 | Implement joint mapping and coordinate validation | Planned |
| 5 | Implement individual cost functions | Planned |
| 6 | Implement MC-MCF network and constraints | Planned |
| 7 | Implement parameter search for \(\alpha,\beta,\gamma\) | Planned |
| 8 | Evaluate
