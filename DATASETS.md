# Datasets — Sources, Access, Citations

All data used in this analysis is publicly accessible. Two sources require HuggingFace authentication (free read token); the rest are open-access. This document catalogs each source, the access mechanism, and the local file path the analysis scripts expect.

---

## 1. CDC NWSS — SARS-CoV-2 Concentration in Wastewater

- **Dataset ID:** `g653-rqe2`
- **URL:** https://data.cdc.gov/Public-Health-Surveillance/NWSS-Public-SARS-CoV-2-Concentration-in-Wastewater/g653-rqe2
- **Update cadence:** weekly (Fridays)
- **Coverage:** July 2020 – present, ~1,500 US sites
- **Access:** public CSV download, no auth
- **Local path:** `data/raw/nwss_concentration.csv`
- **Schema:** `key_plot_id`, `date`, `pcr_conc_lin` (concentration), `normalization` (`flow-population` or `microbial`)

```bash
curl -sL "https://data.cdc.gov/api/views/g653-rqe2/rows.csv?accessType=DOWNLOAD" \
     -o data/raw/nwss_concentration.csv
```

## 2. CDC NWSS — SARS-CoV-2 Wastewater Metric Data

- **Dataset ID:** `2ew6-ywp6`
- **URL:** https://data.cdc.gov/Public-Health-Surveillance/NWSS-Public-SARS-CoV-2-Wastewater-Metric-Data/2ew6-ywp6
- **Coverage:** July 2020 – present
- **Access:** public CSV
- **Local path:** `data/raw/nwss_metric.csv`
- **Schema:** site identifier, jurisdiction, county FIPS, **`population_served`** (used for aggregation weighting), 15-day percent change, percentile metrics

```bash
curl -sL "https://data.cdc.gov/api/views/2ew6-ywp6/rows.csv?accessType=DOWNLOAD" \
     -o data/raw/nwss_metric.csv
```

## 3. CDC NWSS — Influenza A Wastewater

- **Dataset ID:** `ymmh-divb`
- **URL:** https://data.cdc.gov/Public-Health-Surveillance/NWSS-Public-Flu-A-Concentration-in-Wastewater/ymmh-divb (search "Flu A wastewater" on data.cdc.gov)
- **Update cadence:** weekly (Fridays)
- **Coverage:** September 2021 – present, ~1,100 US sites
- **Access:** public CSV
- **Local path:** `data/raw/nwss_flu_a.csv`
- **Schema:** different from COVID — uses `sample_collect_date`, `state_territory`, `pcr_target_flowpop_lin` (flow-population concentration), `pcr_target_mic_lin` (microbial concentration)

```bash
curl -sL "https://data.cdc.gov/api/views/ymmh-divb/rows.csv?accessType=DOWNLOAD" \
     -o data/raw/nwss_flu_a.csv
```

## 4. Google Trends — search-query volume

- **API:** unofficial, via [`pytrends`](https://github.com/GeneralMills/pytrends)
- **Geo scopes:** `US` (national), `US-NV` (Nevada)
- **Time range:** 2021-08-29 → 2025-09-14, weekly cadence
- **Keyword groups:**
  - COVID: "covid symptoms", "loss of taste", "loss of smell", "covid test", "sore throat"
  - Flu: "fever", "cough", "congestion", "runny nose", "body aches"
- **Access:** unauthenticated, but rate-limited and unstable (Google has shut down `pytrends` in the past). Use sparingly.
- **Local path:** `data/raw/trends_nv_us.csv`
- **Pull script:** `notebooks/_pull_trends.py`

## 5. Wikipedia pageviews — Wikimedia REST API

- **API:** https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/...
- **Project:** `en.wikipedia.org`
- **Access type:** `all-access`, agent type `user` (excludes spiders)
- **Granularity:** daily
- **Articles pulled:**
  - `COVID-19`, `Symptoms_of_COVID-19`, `Anosmia`, `Ageusia`
  - `Influenza`, `Common_cold`, `Cough`, `Fever`, `Sore_throat`, `Respiratory_syncytial_virus`
- **Access:** public, no auth, ~100 req/sec rate limit (we use 1 req per 0.1s)
- **Local path:** `data/raw/wikipedia_pageviews.csv`
- **Pull script:** `notebooks/_pull_wikipedia.py`

## 6. CDC ILINet — outpatient influenza-like illness surveillance

- **Source:** CDC FluView, accessed via CMU Delphi epidata API
- **API URL:** https://api.delphi.cmu.edu/epidata/fluview/
- **Regions used:** `nat` (national), `hhs9` (HHS Region 9 — AZ, CA, HI, NV)
- **Time range:** epiweek 202136 → 202538 (Sept 2021 → Sept 2025)
- **Cadence:** weekly
- **Key columns:** `wili` (weighted ILI %), `ili` (raw %), `num_ili`, `num_patients`
- **Access:** public, no auth
- **Local path:** `data/raw/ilinet.csv`
- **Pull script:** `notebooks/_pull_ilinet.py`

## 7. HHS COVID hospital admissions

- **Source:** US Department of Health and Human Services, accessed via CMU Delphi covidcast API
- **API URL:** https://api.delphi.cmu.edu/epidata/covidcast/
- **Source/signal:** `data_source=hhs`, `signal=confirmed_admissions_covid_1d`
- **Geo:** state (`nv`), nation (`us`)
- **Time range:** 2021-09-01 → 2024-04-26 (federal reporting requirements changed mid-2024; data quality degrades after)
- **Cadence:** daily
- **Access:** public, no auth
- **Local path:** `data/raw/covid_ground_truth.csv`
- **Pull script:** `notebooks/_pull_covid_truth.py`

## 8. HHS Influenza hospital admissions

- **Source:** same as above, signal `confirmed_admissions_influenza_1d`
- **Time range:** 2021-09-01 → 2024-04-26
- **Local path:** `data/raw/hhs_flu_admissions.csv`
- **Pull script:** `notebooks/_pull_flu.py` (combined with NWSS Flu A pull)

## 9. WildChat-4.8M — LLM conversation corpus

- **HuggingFace:** [`allenai/WildChat-4.8M`](https://huggingface.co/datasets/allenai/WildChat-4.8M)
- **Authors:** Zhao et al. 2024, NeurIPS Datasets
- **Size:** 3.2M conversations in train split, ~15 GB on disk (parquet)
- **Coverage:** April 9, 2023 → May 2, 2024
- **Schema:** `conversation_hash`, `model`, `timestamp`, `conversation` (list of turns), `language`, `country`, `state`, `hashed_ip`
- **Access:** **gated** — requires HF account, accept dataset terms, and a read token
- **Local path:** `data/raw/wildchat-4.8m/`
- **Pull script:** `src/hf_download.py`

To set up:
```bash
# 1. Create HF account: https://huggingface.co/join
# 2. Accept dataset terms at the URL above
# 3. Create a read token: https://huggingface.co/settings/tokens
# 4. Save token to project (gitignored)
echo "hf_your_token" > .hf_token
python -m src.hf_download
```

## 10. WildChat-1M — older filtered version

- **HuggingFace:** [`allenai/WildChat-1M`](https://huggingface.co/datasets/allenai/WildChat-1M)
- **Size:** 838k non-toxic conversations, ~3 GB
- **Same access requirements as WildChat-4.8M.**
- **Local path:** `data/raw/wildchat-1m/`

We downloaded both for the pilot but only used WildChat-4.8M in the final analysis.

---

## Datasets we tried but did not use in the final analysis

- **LMSYS-Chat-1M** ([`lmsys/lmsys-chat-1m`](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)): 1M conversations from Chatbot Arena, April–August 2023. Access was blocked (gate not approved at submission time) and the time range doesn't overlap any of our seven labeled COVID waves, so this would only have served as a control dataset.
- **Microsoft Copilot health-query data** (Costa-Gomes et al. 2026, *Nature Health*): the dataset class our LLM-prompt pilot most needs — 500k de-identified health-related conversations from January 2026, broad demographics. Not publicly available; analyzed only within Microsoft-controlled systems. Discussed extensively in the paper as evidence that the analytical capability already exists at LLM-provider scale and motivating a system-design proposal for privacy-preserving aggregate releases.

---

## Citations for data sources

- **NWSS data:** CDC National Wastewater Surveillance System. https://www.cdc.gov/nwss/
- **CMU Delphi:** Reinhart, A. et al. (2021). An open repository of real-time COVID-19 indicators. *PNAS*, 118(51), e2111452118. https://api.delphi.cmu.edu/epidata/
- **WildChat:** Zhao, W. et al. (2024). WildChat: 1M ChatGPT interaction logs in the wild. *NeurIPS Datasets and Benchmarks Track*.
- **Costa-Gomes et al. 2026** (Microsoft Copilot data): Costa-Gomes, B. et al. (2026). Public use of a generalist LLM chatbot for health queries. *Nature Health*. arxiv:2604.15331.

Full reference list with DOIs is in `paper/paper.md` Section 8.
