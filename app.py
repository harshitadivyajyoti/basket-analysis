"""Streamlit dashboard.  Run:  streamlit run app.py"""
import streamlit as st, pandas as pd, plotly.express as px
import analysis as A

st.set_page_config(page_title="Assortment Intelligence", layout="wide")
st.title("🛒 Product Assortment Intelligence & Basket Analysis")

@st.cache_data
def get_data(path): return A.load(path)

up = st.sidebar.file_uploader("Upload transactions CSV (optional)", type="csv")
df = pd.read_csv(up) if up else get_data("data/transactions.csv")

st.sidebar.header("Mining parameters")
min_sup = st.sidebar.slider("Min support", 0.002, 0.10, 0.01, 0.002)
min_lift = st.sidebar.slider("Min lift", 1.0, 5.0, 1.2, 0.1)
min_conf = st.sidebar.slider("Min confidence", 0.05, 1.0, 0.2, 0.05)
stores = ["All stores"] + sorted(df.store.unique())
store = st.sidebar.selectbox("Store", stores)
d = df if store == "All stores" else df[df.store == store]

@st.cache_data
def mine(d, s, l, c): return A.mine_rules(d, s, l, c)
freq, rules = mine(d, min_sup, min_lift, min_conf)
pr = A.pair_rules(rules) if len(rules) else rules

c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions", f"{d.transaction_id.nunique():,}")
c2.metric("Products", d["product"].nunique())
c3.metric("Frequent itemsets", len(freq))
c4.metric("Rules found", len(rules))

t1, t2, t3, t4 = st.tabs(["Rules explorer", "Support vs Lift", "Product network/heatmap", "Store assortment"])

with t1:
    st.subheader("Association rules (sorted by lift)")
    st.dataframe(rules.style.format({"support": "{:.3f}", "confidence": "{:.2f}",
                 "lift": "{:.2f}", "leverage": "{:.3f}"}), use_container_width=True)
    st.caption("Support = how common the combo is. Confidence = P(B|A). Lift = confidence / P(B); >1 means real affinity.")
    item = st.selectbox("Recommend for product:", sorted(d["product"].unique()))
    rec = pr[pr.antecedents == item].head(5) if len(pr) else pr
    st.write("Customers who buy **%s** also buy:" % item)
    st.dataframe(rec[["consequents", "confidence", "lift"]] if len(rec) else rec)

with t2:
    if len(pr):
        fig = px.scatter(pr, x="support", y="confidence", size="lift", color="lift",
                         hover_data=["antecedents", "consequents"], color_continuous_scale="Viridis",
                         title="Each dot is a rule: far right = frequent, high colour = truly associated")
        st.plotly_chart(fig, use_container_width=True)
        top = A.classify(pr).head(30)
        st.write("Frequent ≠ associated: rules with lift near 1 are just popular items bought together by chance.")
        st.dataframe(A.classify(pr)[["antecedents", "consequents", "support", "lift", "type"]])
    else: st.info("No rules – lower thresholds.")

with t3:
    if len(pr):
        piv = pr.pivot_table(index="antecedents", columns="consequents", values="lift")
        st.plotly_chart(px.imshow(piv, color_continuous_scale="YlOrRd", aspect="auto",
                        title="Lift heatmap (antecedent → consequent)"), use_container_width=True)

with t4:
    cA, cB = st.columns(2)
    cA.subheader("Store KPIs"); cA.dataframe(A.store_summary(df))
    share = A.store_category_share(df)
    cB.plotly_chart(px.imshow(share, text_auto=True, color_continuous_scale="Blues",
                    title="Category share of items sold (%)"), use_container_width=True)
    sr = A.store_rules(df, min_sup, min_lift)
    if len(sr):
        st.subheader("Top rule per store")
        st.dataframe(sr.groupby("store").head(3))
        s2 = st.selectbox("Assortment gap analysis for store:", sorted(df.store.unique()))
        st.write("High-affinity pairs where the add-on item is under-sold here → promote / co-locate / stock more:")
        st.dataframe(A.assortment_gaps(df, sr, s2))
