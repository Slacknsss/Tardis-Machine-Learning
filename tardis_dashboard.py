import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime

df = pd.read_csv("dataset_cleaned.csv", sep=";")

model = joblib.load("model.joblib")
le_dep = joblib.load("le_dep.joblib")
le_arr = joblib.load("le_arr.joblib")

delay_col = "Retard moyen de tous les trains à l'arrivée"

page = st.sidebar.radio("Navigation", ["Graphiques", "Prédiction"])

# PAGE GRAPHIQUES (EDA)
if page == "Graphiques":
    st.title("Exploration des données")

    # Tendance mensuelle
    if "Date" in df.columns and delay_col in df.columns:
        st.subheader("Tendance mensuelle des retards")

        tmp = df.copy()
        tmp["Date"] = pd.to_datetime(tmp["Date"], errors="coerce")
        tmp["month_period"] = tmp["Date"].dt.to_period("M")

        series = (
            tmp.groupby("month_period")[delay_col]
            .mean()
            .dropna()
        )

        series.index = series.index.to_timestamp()

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(series.index, series.values)

        ax.set_title("Tendance mensuelle des retards")
        ax.set_xlabel("Mois")
        ax.set_ylabel("Retard moyen (min)")

        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        fig.autofmt_xdate()
        st.pyplot(fig)

    # Top gares
    if delay_col in df.columns and "Gare d'arrivée" in df.columns:
        st.subheader("Top 10 gares les plus retardées")

        by_station = (
            df.groupby("Gare d'arrivée")[delay_col]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )

        fig, ax = plt.subplots()
        ax.bar(by_station.index, by_station.values)
        ax.set_title("Top 10 gares (retard moyen)")
        ax.set_xlabel("Gare d'arrivée")
        ax.set_ylabel("Retard moyen (min)")
        plt.xticks(rotation=60, ha="right")
        st.pyplot(fig)

    # Matrice départ / arrivée
    st.subheader("Matrice départ / arrivée (retard moyen)")

    if delay_col in df.columns:
        top_dep = df["Gare de départ"].value_counts().head(8).index
        top_arr = df["Gare d'arrivée"].value_counts().head(8).index

        pivot = (
            df[df["Gare de départ"].isin(top_dep) & df["Gare d'arrivée"].isin(top_arr)]
            .pivot_table(index="Gare de départ", columns="Gare d'arrivée", values=delay_col, aggfunc="mean")
        )

        M = pivot.astype(float).fillna(0).to_numpy()

        fig, ax = plt.subplots()
        im = ax.imshow(M, aspect="auto")
        ax.set_title("Retard moyen (top départ / arrivée)")
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns, rotation=60, ha="right")
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels(pivot.index)
        fig.colorbar(im, ax=ax, label="Retard moyen (min)")
        st.pyplot(fig)

# PAGE PRÉDICTION
else:
    st.title("Prédiction de retard")

    gares_dep = sorted(df["Gare de départ"].dropna().unique())
    gares_arr = sorted(df["Gare d'arrivée"].dropna().unique())

    gare_dep = st.selectbox("Gare de départ", gares_dep)
    gare_arr = st.selectbox("Gare d'arrivée", gares_arr)

    mois = st.slider("Mois", 1, 12, datetime.datetime.now().month)

    # durée moyenne par liaison
    if "Durée moyenne du trajet" in df.columns:
        df_route = df[
            (df["Gare de départ"] == gare_dep) & (df["Gare d'arrivée"] == gare_arr)
        ]

        if len(df_route) > 0:
            duree_moy = df_route["Durée moyenne du trajet"].mean()
        else:
            duree_moy = df["Durée moyenne du trajet"].mean()

    else:
        duree_moy = 0
        st.write("Durée moyenne indisponible")


    dep_enc = le_dep.transform([gare_dep])[0]
    arr_enc = le_arr.transform([gare_arr])[0]

    X_pred = pd.DataFrame(
        [[dep_enc, arr_enc, mois, duree_moy]],
        columns=[
            "Gare_depart_enc",
            "Gare_arrivee_enc",
            "Mois",
            "Durée moyenne du trajet",
        ],
    )

    try:
        pred = model.predict(X_pred)[0]
        st.success(f"Retard prédit : {pred:.2f} minutes")
    except Exception as e:
        st.error(f"Erreur de prédiction : {e}")
        st.write(
            "Colonnes attendues :",
            model.feature_names_in_ if hasattr(model, "feature_names_in_") else "inconnues",
        )