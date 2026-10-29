"""Generate synthetic retail transactions with planted product affinities
and store-specific behaviour. Output: data/transactions.csv
Columns: transaction_id, store, product, category
"""
import numpy as np, pandas as pd
rng = np.random.default_rng(42)

PRODUCTS = {
    "Dairy": ["Milk", "Butter", "Cheese", "Yogurt"],
    "Bakery": ["Bread", "Bagel", "Croissant"],
    "Produce": ["Banana", "Apple", "Tomato", "Onion", "Lettuce"],
    "Snacks": ["Chips", "Cookies", "Chocolate", "Popcorn"],
    "Beverages": ["Cola", "Juice", "Coffee", "Tea", "Beer"],
    "Pantry": ["Pasta", "Pasta Sauce", "Rice", "Cereal", "Eggs"],
    "Household": ["Detergent", "Tissue", "Soap"],
}
CAT = {p: c for c, ps in PRODUCTS.items() for p in ps}
ALL = list(CAT)

# planted affinities: (antecedent, consequent, probability consequent added)
AFFINITY = [("Pasta", "Pasta Sauce", .75), ("Bread", "Butter", .55),
            ("Chips", "Cola", .50), ("Cereal", "Milk", .60),
            ("Coffee", "Cookies", .40), ("Beer", "Chips", .45),
            ("Tomato", "Onion", .45), ("Bagel", "Cheese", .40)]

# stores: name -> favoured categories (boosts base popularity)
STORES = {"Downtown": ["Beverages", "Snacks"], "Suburb": ["Dairy", "Household", "Pantry"],
          "Campus": ["Snacks", "Beverages", "Bakery"], "Market St": ["Produce", "Bakery"]}

def make(n=20000):
    rows = []
    stores = list(STORES)
    for tid in range(1, n + 1):
        s = stores[rng.integers(len(stores))]
        w = np.array([1.0 + (1.5 if CAT[p] in STORES[s] else 0) for p in ALL])
        w *= rng.uniform(.5, 1.5, len(ALL)) ** 0  # keep deterministic shape
        w /= w.sum()
        k = max(1, int(rng.poisson(3)))
        basket = set(rng.choice(ALL, size=min(k, len(ALL)), replace=False, p=w))
        for a, c, p in AFFINITY:
            if a in basket and rng.random() < p:
                basket.add(c)
        rows += [(tid, s, p, CAT[p]) for p in basket]
    return pd.DataFrame(rows, columns=["transaction_id", "store", "product", "category"])

if __name__ == "__main__":
    df = make()
    df.to_csv("data/transactions.csv", index=False)
    print(f"Saved {df.transaction_id.nunique()} transactions, {len(df)} rows -> data/transactions.csv")
