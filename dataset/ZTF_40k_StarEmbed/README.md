---
license: mit
dataset_info:
  features:
  - name: sourceid
    dtype: string
  - name: bands_data
    struct:
    - name: g
      struct:
      - name: feat_dynamic_real
        list: float64
      - name: length
        dtype: int64
      - name: mjd
        list: float64
      - name: past_feat_dynamic_real
        list: float64
      - name: target
        list: float64
    - name: i
      struct:
      - name: feat_dynamic_real
        list: float64
      - name: length
        dtype: int64
      - name: mjd
        list: float64
      - name: past_feat_dynamic_real
        list: float64
      - name: target
        list: float64
    - name: r
      struct:
      - name: feat_dynamic_real
        list: float64
      - name: length
        dtype: int64
      - name: mjd
        list: float64
      - name: past_feat_dynamic_real
        list: float64
      - name: target
        list: float64
  - name: period
    dtype: float64
  - name: class_str
    dtype: string
  - name: ra
    dtype: float64
  - name: dec
    dtype: float64
  splits:
  - name: train
    num_bytes: 874152765
    num_examples: 29047
  - name: validation
    num_bytes: 125721375
    num_examples: 4150
  - name: test
    num_bytes: 247388574
    num_examples: 8301
  - name: anom
    num_bytes: 32708516
    num_examples: 1087
  download_size: 557951471
  dataset_size: 1279971230
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
  - split: anom
    path: data/anom-*
---

# StarEmbed Benchmark Data set

The StarEmbed benchmark data set is available here in huggingface datasets format.

The ~40,000 multi-band ZTF light curves are available at `StarEmbed/data`.
The train/validation/test splits are included alongside the additional anom split used for the OOD benchmarking.


## Available columns

Each star in data set has the following fields:

**Top-level columns**
| Column       | Type                               | Description                                                                                               |
| ------------ | ---------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `sourceid`   | string                             | Catalina Surveys DR1 (CSDR1) source id of the variable star (e.g. `CSS_J082956.4-044426`)                |
| `bands_data` | dict / struct (keys `g`, `i`, `r`) | per-band light curves; each band is a struct (or `null` if absent) holding the arrays in the table below.|
| `period`     | float64                            | catalog period of the variable star, in days (range ≈ 0.13 – 885)                                        |
| `class_str`  | string                             | variable-star class label; one of `EW`, `EA`, `RRab`, `RRc`, `RRd`, `RS CVn`, `LPV`                      |
| `ra`         | float64                            | right ascension (J2000) in decimal degrees                                                               |
| `dec`        | float64                            | declination (J2000) in decimal degrees                                                                   |

**Per-band fields inside each band of `bands_data` (`g` / `r` / `i`)**
| Field                    | Type           | Description                                                          |
| ------------------------ | -------------- | ------------------------------------------------------------------- |
| `target`                 | list of floats | magnitude, measure of brightness (AB system) — the raw light curve  |
| `past_feat_dynamic_real` | list of floats | mag_err, 1σ uncertainty on magnitude, aligned with `target`         |
| `feat_dynamic_real`      | list of floats | delta_t, time gap (days) between each two consecutive observations  |
| `mjd`                    | list of floats | observation time in Modified Julian Date, aligned with `target`     |
| `length`                 | int64          | number of observations in this band's light curve                   |

All four splits (train, validation, test, anom) share the identical schema described above. The train/validation/test splits contain the seven in-distribution classes listed under `class_str`, whereas the anom split is an anomaly-detection holdout whose `class_str` values are a disjoint set of out-of-distribution classes — `Beta_Lyrae`, `Blazhko`, `ACEP`, `Cep-II`, `HADS`, `LADS`, `ELL`, `Hump`, `PCEB`, `EA_UP` — that never appear in training.

---

## **Citation**
This work was accepted to the main track of ICML 2026 as well as the AI4Physics workshop. If you use or refer to this dataset please cite our work.
```
@article{StarEmbed,
       author = {{Li}, Weijian and {Chen}, Hong-Yu and {Rehemtulla}, Nabeel and {Shah}, Ved G. and {Wu}, Dennis and {Kim}, Dongho and {Lin}, Qinjie and {Miller}, Adam A. and {Liu}, Han},
        title = "{StarEmbed: Benchmarking Time Series Foundation Models on Astronomical Observations of Variable Stars}",
      journal = {arXiv e-prints},
     keywords = {Solar and Stellar Astrophysics, Instrumentation and Methods for Astrophysics, Artificial Intelligence},
         year = 2025,
        month = oct,
          eid = {arXiv:2510.06200},
        pages = {arXiv:2510.06200},
          doi = {10.48550/arXiv.2510.06200},
archivePrefix = {arXiv},
       eprint = {2510.06200},
 primaryClass = {astro-ph.SR},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025arXiv251006200L},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

The StarEmbed data set is curated from ZTF observations and the Catalina Surveys Periodic Variable Star Catalog. If you use the StarEmbed data please also cite the works below.
```
@article{bellm2018zwicky,
  title={The Zwicky Transient Facility: system overview, performance, and first results},
  author={Bellm, Eric C and Kulkarni, Shrinivas R and Graham, Matthew J and Dekany, Richard and Smith, Roger M and Riddle, Reed and Masci, Frank J and Helou, George and Prince, Thomas A and Adams, Scott M and others},
  journal={Publications of the Astronomical Society of the Pacific},
  volume={131},
  number={995},
  pages={018002},
  year={2018},
  publisher={IOP Publishing}
}

@article{drake2014catalina,
  title={The catalina surveys periodic variable star catalog},
  author={Drake, AJ and Graham, MJ and Djorgovski, SG and Catelan, M and Mahabal, AA and Torrealba, G and Garc{\'\i}a-{\'A}lvarez, D and Donalek, C and Prieto, JL and Williams, R and others},
  journal={The Astrophysical Journal Supplement Series},
  volume={213},
  number={1},
  pages={9},
  year={2014},
  publisher={IOP Publishing}
}025}
}
```