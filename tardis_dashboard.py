import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(
    page_title="TARDIS - SNCF Delay Predictor",
    layout="wide"
)

st.title("TARDIS")
st.subheader("Analyse et prédiction des retards SNCF")


@st.cache_data
def load_data():
    df = pd.read_csv("dataset_clean.csv", sep=";")
    df.columns = df.columns.str.strip()
    return df


@st.cache_resource
def load_model():
    return joblib.load("model.joblib")


df = load_data()
model = load_model()

delay_col = "Retard moyen de tous les trains à l'arrivée"


page = st.sidebar.selectbox(
    "Navigation",
    ["Analyse Globale", "Analyse par Liaison", "Prédiction"]
)


if page == "Analyse Globale":

    st.header("Vue globale du réseau")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 10 des liaisons les plus en retard")

        top_routes = (
            df.groupby(["Gare de départ", "Gare d'arrivée"])[delay_col]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        top_routes["Liaison"] = (
            top_routes["Gare de départ"] + " -> " +
            top_routes["Gare d'arrivée"]
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(
            data=top_routes,
            x=delay_col,
            y="Liaison",
            ax=ax
        )
        ax.set_xlabel("Retard moyen (minutes)")
        ax.set_ylabel("Liaison")
        st.pyplot(fig)

    with col2:
        st.subheader("Evolution mensuelle des retards")

        monthly = (
            df.groupby("month")[delay_col]
            .mean()
            .reset_index()
        )

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.lineplot(
            data=monthly,
            x="month",
            y=delay_col,
            marker="o",
            ax=ax2
        )
        ax2.set_xlabel("Mois")
        ax2.set_ylabel("Retard moyen (minutes)")
        st.pyplot(fig2)

    st.subheader("Répartition moyenne des causes de retard")

    cause_cols = [c for c in df.columns if "Prct retard" in c]
    cause_means = df[cause_cols].mean().sort_values(ascending=False)

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    sns.barplot(
        x=cause_means.values,
        y=cause_means.index,
        ax=ax3
    )
    ax3.set_xlabel("Pourcentage moyen (%)")
    ax3.set_ylabel("Cause")
    st.pyplot(fig3)


elif page == "Analyse par Liaison":

    st.header("Analyse personnalisée d'une liaison")

    departure = st.selectbox(
        "Gare de départ",
        sorted(df["Gare de départ"].unique())
    )

    arrival = st.selectbox(
        "Gare d'arrivée",
        sorted(df["Gare d'arrivée"].unique())
    )

    filtered = df[
        (df["Gare de départ"] == departure) &
        (df["Gare d'arrivée"] == arrival)
    ]

    if filtered.empty:
        st.warning("Aucune donnée disponible pour cette liaison.")
    else:

        avg_delay = filtered[delay_col].mean()

        cancel_rate = (
            filtered["Nombre de trains annulés"].sum() /
            filtered["Nombre de circulations prévues"].sum()
        ) * 100

        col1, col2 = st.columns(2)

        col1.metric("Retard moyen", f"{avg_delay:.2f} min")
        col2.metric("Taux d'annulation", f"{cancel_rate:.2f} %")

        st.subheader("Evolution mensuelle pour cette liaison")

        monthly_filtered = (
            filtered.groupby("month")[delay_col]
            .mean()
            .reset_index()
        )

        fig4, ax4 = plt.subplots()
        sns.lineplot(
            data=monthly_filtered,
            x="month",
            y=delay_col,
            marker="o",
            ax=ax4
        )
        ax4.set_xlabel("Mois")
        ax4.set_ylabel("Retard moyen (minutes)")
        st.pyplot(fig4)


elif page == "Prédiction":

    st.header("Prédiction du retard estimé")

    departure = st.selectbox(
        "Gare de départ",
        sorted(df["Gare de départ"].unique())
    )

    arrival = st.selectbox(
        "Gare d'arrivée",
        sorted(df["Gare d'arrivée"].unique())
    )

    month = st.slider("Mois", 1, 12, 6)

    duration = st.number_input(
        "Durée moyenne du trajet (minutes)",
        min_value=0.0,
        value=120.0
    )

    trains = st.number_input(
        "Nombre de trains programmés",
        min_value=0,
        value=10
    )

    if st.button("Estimer le retard"):

        input_df = pd.DataFrame([{
            "Gare de départ": departure,
            "Gare d'arrivée": arrival,
            "Mois": month,
            "Durée moyenne du trajet": duration,
            "Nombre de trains programmés": trains
        }])

        prediction = model.predict(input_df)[0]

        st.success(f"Retard estimé : {prediction:.2f} minutes")

        if prediction < 5:
            st.write("Risque faible de retard.")
        elif prediction < 15:
            st.write("Risque modéré de retard.")
        else:
            st.write("Risque élevé de retard.")