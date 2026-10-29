"""Core analytics: basket matrix, association rules, store-level assortment."""
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def load(path="data/transactions.csv"):
    return pd.read_csv(path)

def basket_matrix(df):
    """One row per transaction, one boolean column per product."""
    return pd.crosstab(df.transaction_id, df["product"]).astype(bool)

def mine_rules(df, min_support=0.01, min_lift=1.0, min_conf=0.1):
    """Apriori -> rules with support, confidence, lift."""
    basket = basket_matrix(df)
    freq = apriori(basket, min_support=min_support, use_colnames=True)
    if freq.empty:
        return freq, pd.DataFrame()
    rules = association_rules(freq, metric="lift", min_threshold=min_lift)
    rules = rules[rules.confidence >= min_conf].copy()
    rules["antecedents"] = rules.antecedents.apply(lambda s: ", ".join(sorted(s)))
    rules["consequents"] = rules.consequents.apply(lambda s: ", ".join(sorted(s)))
    cols = ["antecedents", "consequents", "support", "confidence", "lift", "leverage"]
    return freq, rules[cols].sort_values("lift", ascending=False).reset_index(drop=True)

def pair_rules(rules):
    """Keep only 1-item -> 1-item rules (easier to read)."""
    m = ~rules.antecedents.str.contains(",") & ~rules.consequents.str.contains(",")
    return rules[m].reset_index(drop=True)

def classify(rules, lift_cut=1.2):
    """Frequent-but-independent vs genuinely associated."""
    r = rules.copy()
    r["type"] = pd.cut(r.lift, [0, .95, lift_cut, 1e9],
                       labels=["Negative", "Coincidental (lift~1)", "Meaningful"])
    return r

def store_summary(df):
    """Per-store KPIs."""
    g = df.groupby("store")
    return pd.DataFrame({
        "transactions": g.transaction_id.nunique(),
        "unique_products": g["product"].nunique(),
        "avg_basket_size": g.size() / g.transaction_id.nunique(),
    }).round(2)

def store_category_share(df):
    t = pd.crosstab(df.store, df.category)
    return (t.div(t.sum(1), axis=0) * 100).round(1)

def store_rules(df, min_support=0.01, min_lift=1.2):
    """Mine rules separately per store, return stacked frame."""
    out = []
    for s, d in df.groupby("store"):
        _, r = mine_rules(d, min_support, min_lift)
        if len(r):
            r = pair_rules(r); r.insert(0, "store", s); out.append(r)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

def assortment_gaps(df, rules_store, store, top=10):
    """Rules strong in a store whose consequent is a weak seller there -> opportunity."""
    d = df[df.store == store]
    share = d.groupby("product").transaction_id.nunique() / d.transaction_id.nunique()
    r = rules_store[rules_store.store == store].copy()
    r["consequent_store_support"] = r.consequents.map(share)
    r["gap_score"] = r.lift * r.confidence / (r.consequent_store_support + 1e-9)
    return r.sort_values("gap_score", ascending=False).head(top)
