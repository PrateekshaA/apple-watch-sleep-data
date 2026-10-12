# BIDSleep Apple Watch Dataset

## Overview

This dataset contains multi-night sleep recordings collected from Apple Watch devices across 47 subjects. Each subject includes between 3 and 7 nights of data. For each night, the dataset provides accelerometer signals, heart rate measurements, and sleep-stage annotations.

The dataset is designed to support research in wearable-based sleep staging, physiological signal processing, and multimodal time-series modeling.

---

## Dataset Organization

The dataset is organized hierarchically into subject-level and night-level folders:

```
<base>/
├── Bidslab00/
│   ├── 1/
│   │   ├── motion.csv
│   │   ├── hr.csv
│   │   └── labels.mat
│   ├── 2/
│   ├── ...
│   └── 7/
├── Bidslab01/
├── ...
├── Bidslab68/
```

* **Subject-level folders (`BidslabXX`)**
  Each folder corresponds to a unique subject.

* **Night-level folders (`1`, `2`, ..., `k`)**
  Each subfolder represents one full-night recording session for that subject.

* Night indices are ordered chronologically.

Each night folder is **self-contained** and can be used independently for analysis.

---

## Data Files (Per Night)

Each night folder contains three files:

### 1. `motion.csv`

Three-axis accelerometer data recorded by the Apple Watch:

* Columns: `Timestamp`, `x`, `y`, `z`
* `Timestamp`: **Unix time in seconds (floating-point, sub-second precision)**
* `x`, `y`, `z`: three-axis accelerometer measurements (in units of gravitational acceleration, g)  
* Captures wrist motion during sleep  
 
---

### 2. `hr.csv`

Instantaneous heart rate (IHR) values:

* Format: no header row (two columns)

  - Column 1: timestamp (Unix time in seconds, floating-point, sub-second precision)  
  - Column 2: hr (heart rate in beats per minute, bpm)

* Source: Apple Watch PPG sensor via HealthKit  
* Sampling rate: approximately **0.2 Hz**

* Note: This file does not include a header row. Column names are provided here for interpretation only.

---

### 3. `labels.mat`

Contains sleep-stage annotations and metadata:

* `recStart`
  Recording start timestamp (originally in Unix time, stored as human-readable time in U.S. Eastern Time)

* `dreem_label`
  Automated sleep-stage annotations exported from the Dreem device

* `expert_label`
  Manually reviewed and corrected sleep-stage annotations by a sleep expert

---

## Sleep Stage Encoding

Sleep stages are encoded as integers:

* Wake = 0
* N1 = 1
* N2 = 2
* N3 = 3
* REM = 4
* Unknown = 5

---

## Timestamp Alignment

The `motion.csv` and `hr.csv` timestamps are expressed in **absolute Unix time (seconds)**.
The variable `recStart` in `labels.mat` represents the recording start time and serves as the reference point for label alignment.

To align physiological signals with sleep stage labels:

1. Convert `recStart` to Unix time (if needed).

2. Each sleep stage label corresponds to a **30-second epoch** relative to `recStart`:

   * 1st epoch:

     ```
     [recStart, recStart + 30 s)
     ```
   * k-th epoch:

     ```
     [recStart + 30 × (k − 1), recStart + 30 × k)
     ```

3. For any timestamp `t` in `motion.csv` or `hr.csv`, the corresponding sleep stage index `k` is:

```
k = floor((t − recStart) / 30) + 1
```

This enables direct mapping of heart rate and accelerometry samples to their corresponding sleep stage epochs.

---

## Relationship Between Subjects and Nights

* Each subject (`BidslabXX`) may have a different number of recorded nights (typically 3–7).
* Each night folder corresponds to a **separate recording session**.
* All data within a night folder are temporally aligned to the same `recStart`.

---

## Notes

* Timestamps in `motion.csv` and `hr.csv` are provided in **Unix time (UTC)**.
* `recStart` is stored in **U.S. Eastern Time (ET)** but can be converted back to Unix time for alignment.
* Data were collected in free-living conditions using consumer wearable devices.
* Minor irregular sampling may occur due to device constraints.

---

## Suggested Use Cases

* Sleep stage classification
* Multimodal time-series modeling (heart rate + motion)
* Domain adaptation across subjects or devices
* Benchmarking wearable sleep algorithms

---

