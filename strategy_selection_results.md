# Strategy Selection Results: UCR Benchmark Datasets

This document compiles the segmentation routing strategy selections (CPD vs. Fixed windowing) dynamically computed across the UCR benchmark datasets under the **Complexity-Normalized Same-Budget Adaptive Router**.

---

## 📊 Summary Counts

* **Fixed Windowing**: **11 datasets**
* **CPD Variable Windowing**: **9 datasets**

---

## 📋 Detailed Strategy Selection Table

Below is the complete list of automatic routing strategies selected on the training splits and their corresponding partition sizes:

| Dataset | Selected Strategy | Segmentation Detail | Decision / Outcome |
| :--- | :---: | :---: | :--- |
| **`GunPoint`** | **`cpd`** | cpd(1.5) | Routed to CPD (10D Acc win/loss: +0.0267) |
| **`Coffee`** | **`fixed`** | fixed(28) | Routed to Fixed (10D Acc win/loss: +0.0000) |
| **`ArrowHead`** | **`fixed`** | fixed(25) | Routed to Fixed (10D Acc win/loss: +0.0229) |
| **`ECG200`** | **`cpd`** | cpd(1.5) | Routed to CPD (10D Acc win/loss: -0.0800) |
| **`Chinatown`** | **`fixed`** | fixed(10) | Routed to Fixed (10D Acc win/loss: -0.0117) |
| **`ItalyPowerDemand`** | **`cpd`** | cpd(1.5) | Routed to CPD (10D Acc win/loss: -0.0486) |
| **`SonyAIBORobotSurface1`** | **`fixed`** | fixed(10) | Routed to Fixed (10D Acc win/loss: +0.0283) |
| **`TwoLeadECG`** | **`fixed`** | fixed(10) | Routed to Fixed (10D Acc win/loss: +0.0421) |
| **`ECGFiveDays`** | **`fixed`** | fixed(13) | Routed to Fixed (10D Acc win/loss: -0.0105) |
| **`MoteStrain`** | **`fixed`** | fixed(10) | Routed to Fixed (10D Acc win/loss: -0.0016) |
| **`Beef`** | **`cpd`** | cpd(1.5) | Routed to CPD (10D Acc win/loss: -0.0333) |
| **`OliveOil`** | **`cpd`** | cpd(1.5) | Routed to CPD (10D Acc win/loss: -0.0333) |
| **`Meat`** | **`cpd`** | cpd(1.5) | Routed to CPD (10D Acc win/loss: -0.1333) |
| **`BeetleFly`** | **`fixed`** | fixed(51) | Routed to Fixed (10D Acc win/loss: -0.2000) |
| **`BirdChicken`** | **`fixed`** | fixed(51) | Routed to Fixed (10D Acc win/loss: +0.0000) |
| **`FaceFour`** | **`fixed`** | fixed(35) | Routed to Fixed (10D Acc win/loss: +0.0000) |
| **`SyntheticControl`** | **`fixed`** | fixed(10) | Routed to Fixed (10D Acc win/loss: +0.0067) |
| **`CBF`** | **`cpd`** | cpd(1.5) | Routed to CPD (10D Acc win/loss: -0.0656) |
| **`TwoPatterns`** | **`cpd`** | cpd(1.5) | Routed to CPD (10D Acc win/loss: -0.1985) |
| **`Wafer`** | **`cpd`** | cpd(1.5) | Routed to CPD (10D Acc win/loss: +0.0055) |
