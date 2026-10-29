# basket-analysis

# Product Assortment Intelligence & Basket Analysis

Association-rule mining on retail transactions to find product affinities, separate genuine associations from merely popular items, and compare assortment across stores. Includes an interactive Streamlit dashboard.

## Features
- **Basket mining:** Apriori on a transaction x product matrix (mlxtend)
- **Rule metrics:** support, confidence, lift, leverage
- **Frequent vs meaningful:** rules labelled Meaningful / Coincidental / Negative by lift
- **Store-level analysis:** per-store rules, KPIs, category share, and an assortment-gap score
- **Dashboard:** adjustable thresholds, rule explorer, product recommender, support-vs-lift scatter, lift heatmap

## Project structure
```
generate_data.py   synthetic transactions with planted affinities and store biases
analysis.py        basket matrix, rule mining, classification, store analytics
run_analysis.py    CLI summary of top rules and store gaps
app.py             Streamlit dashboard
data/              transactions.csv (generated)
```

## Setup
```bash
python -m venv venv
venv\Scripts\activate          # Windows  (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
python generate_data.py
python run_analysis.py
python -m streamlit run app.py
```
Requires Python 3.9+.

## Using your own data
CSV with columns: `transaction_id, store, product, category`.
Upload it in the dashboard sidebar or replace `data/transactions.csv`.

## Metrics
- **Support:** share of baskets containing the itemset
- **Confidence:** P(B | A)
- **Lift:** confidence / P(B). Values above 1 mean A and B appear together more than chance.
- **Gap score:** lift x confidence / consequent's support in a store. High means a strong pair whose add-on item is under-sold there.

## Limitations
- Demo data is synthetic, with affinities planted deliberately
- Apriori is slow on very large catalogs (see Roadmap)
- Rules show correlation, not causal uplift

## Roadmap
- FP-Growth benchmark against Apriori
- Store clustering by category mix
- Validation on a real dataset

<img width="1897" height="867" alt="image" src="https://github.com/user-attachments/assets/72dcfd19-89bf-4ec8-b36d-4d65deda6bb2" />
<img width="1348" height="625" alt="image" src="https://github.com/user-attachments/assets/26a2c2d4-bef8-452a-8a3d-512edb0acac7" />
<img width="1387" height="652" alt="image" src="https://github.com/user-attachments/assets/da13b492-c315-4ee0-b0fb-f774a795399b" />
<img width="1912" height="926" alt="image" src="https://github.com/user-attachments/assets/7bdb594a-133d-4301-a68a-2396b379621d" />



