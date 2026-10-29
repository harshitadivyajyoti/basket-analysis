import analysis as A
df = A.load()
freq, rules = A.mine_rules(df, 0.01, 1.2, 0.2)
pr = A.pair_rules(rules)
print("Top rules by lift:\n", pr.head(10).round(3).to_string(index=False))
print("\nStore KPIs:\n", A.store_summary(df))
sr = A.store_rules(df, 0.01, 1.2)
print("\nGap analysis (Campus):\n", A.assortment_gaps(df, sr, "Campus", 5).round(3).to_string(index=False))
