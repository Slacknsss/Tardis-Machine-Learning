import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="TARDIS | SNCF Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("cleaned_dataset.csv", sep=";")
        df.columns = df.columns.str.strip()
        
        colonnes_a_convertir = ["Nombre de trains annulés", "Nombre de circulations prévues"]
        for col in colonnes_a_convertir:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        
        if 'Mois' not in df.columns:
            if 'month' in df.columns:
                df = df.rename(columns={'month': 'Mois'})
            elif 'Date' in df.columns:
                df['Date'] = df['Date'].astype(str).str.replace('/', '-').str.replace(' ', '-')
                df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
                df['Mois'] = df['Date'].dt.month
                
        return df
    except FileNotFoundError:
        st.error("Fichier 'cleaned_dataset.csv' introuvable. Lancez le notebook EDA d'abord !")
        st.stop()

@st.cache_resource
def load_model_and_encoders():
    try:
        model = joblib.load("model.joblib")
        le_dep = joblib.load("le_dep.joblib")
        le_arr = joblib.load("le_arr.joblib")
        return model, le_dep, le_arr
    except FileNotFoundError:
        st.error("Modèles IA introuvables. Lancez le notebook de Modélisation d'abord !")
        st.stop()

df = load_data()
model, le_dep, le_arr = load_model_and_encoders()

delay_col = "Retard moyen de tous les trains à l'arrivée"

st.sidebar.title("TARDIS Project")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["Analyse Globale", "Analyse par Liaison", "Prédiction IA"]
)
st.sidebar.markdown("---")
st.sidebar.caption("Développé par le Groupe 3 (Simon, Jules, Raph)")

if page == "Analyse Globale":
    st.title("Vue Globale du Réseau SNCF")
    st.markdown("Analyse des retards sur l'ensemble des liaisons TGV étudiées.")

    st.markdown("### Chiffres Clés")
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Liaisons étudiées", f"{len(df.groupby(['Gare de départ', 'Gare d\'arrivée']))}")
    kpi2.metric("Retard Moyen National", f"{df[delay_col].mean():.2f} min")
    
    if "Nombre de trains annulés" in df.columns and "Nombre de circulations prévues" in df.columns:
        taux_annul = (df["Nombre de trains annulés"].sum() / df["Nombre de circulations prévues"].sum()) * 100
        kpi3.metric("Taux d'annulation global", f"{taux_annul:.2f} %")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 10 des pires liaisons")
        top_routes = df.groupby(["Gare de départ", "Gare d'arrivée"])[delay_col].mean().sort_values(ascending=False).head(10).reset_index()
        top_routes["Liaison"] = top_routes["Gare de départ"] + " -> " + top_routes["Gare d'arrivée"]

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=top_routes, x=delay_col, y="Liaison", palette="Reds_r", ax=ax)
        ax.set_xlabel("Retard moyen (minutes)")
        ax.set_ylabel("")
        st.pyplot(fig)

    with col2:
        st.subheader("Saisonnalité des retards")
        monthly = df.groupby("Mois")[delay_col].mean().reset_index()

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.lineplot(data=monthly, x="Mois", y=delay_col, marker="o", color="#1f77b4", linewidth=2.5, ax=ax2)
        ax2.set_xticks(range(1, 13))
        ax2.set_xlabel("Mois (1 = Janvier)")
        ax2.set_ylabel("Retard moyen (minutes)")
        ax2.grid(True, linestyle="--", alpha=0.6)
        st.pyplot(fig2)

elif page == "Analyse par Liaison":
    st.title("Analyse Personnalisée")
    st.markdown("Plongez dans les détails d'un trajet spécifique pour comprendre ses performances.")

    col_dep, col_arr = st.columns(2)
    with col_dep:
        departure = st.selectbox("Sélectionnez la gare de départ", sorted(df["Gare de départ"].unique()))
    
    gares_arrivee_possibles = df[df["Gare de départ"] == departure]["Gare d'arrivée"].unique()
    
    with col_arr:
        arrival = st.selectbox("Sélectionnez la gare d'arrivée", sorted(gares_arrivee_possibles))

    filtered = df[(df["Gare de départ"] == departure) & (df["Gare d'arrivée"] == arrival)]

    if filtered.empty:
        st.warning("Aucune donnée SNCF disponible pour ce trajet direct.")
    else:
        st.markdown("### Statistiques du trajet")
        avg_delay = filtered[delay_col].mean()
        national_avg = df[delay_col].mean()
        
        diff_moyenne = avg_delay - national_avg

        c1, c2, c3 = st.columns(3)
        c1.metric("Retard moyen sur ce trajet", f"{avg_delay:.2f} min", delta=f"{diff_moyenne:.2f} min vs National", delta_color="inverse")
        
        if "Nombre de trains annulés" in filtered.columns and "Nombre de circulations prévues" in filtered.columns:
            total_prevus = filtered["Nombre de circulations prévues"].sum()
            if total_prevus > 0:
                cancel_rate = (filtered["Nombre de trains annulés"].sum() / total_prevus) * 100
            else:
                cancel_rate = 0.0
            c2.metric("Taux d'annulation", f"{cancel_rate:.2f} %")
            
        c3.metric("Nombre de voyages enregistrés", f"{len(filtered)} mois de données")

        st.markdown("---")
        st.subheader("Historique des retards pour ce trajet")
        monthly_filtered = filtered.groupby("Mois")[delay_col].mean().reset_index()
        fig4, ax4 = plt.subplots(figsize=(10, 4))
        sns.barplot(data=monthly_filtered, x="Mois", y=delay_col, palette="Blues", ax=ax4)
        ax4.set_xlabel("Mois")
        ax4.set_ylabel("Retard (min)")
        st.pyplot(fig4)

elif page == "Prédiction IA":
    st.title("Le Prédicteur TARDIS")
    st.markdown("Notre modèle d'Intelligence Artificielle va estimer votre temps de retard.")

    with st.form("prediction_form"):
        st.subheader("Détails de votre voyage")
        
        c1, c2 = st.columns(2)
        with c1:
            departure = st.selectbox("Gare de départ", le_dep.classes_)
            month = st.slider("Mois du voyage (1=Jan, 12=Déc)", 1, 12, 6)
        with c2:
            arrival = st.selectbox("Gare d'arrivée", le_arr.classes_)
            duration = st.number_input("Durée théorique du trajet (en minutes)", min_value=10, max_value=1500, value=120)

        submit_button = st.form_submit_button(label="Lancer la prédiction", use_container_width=True)

    if submit_button:
        with st.spinner("L'IA analyse les données historiques..."):
            try:
                dep_enc = le_dep.transform([departure])[0]
                arr_enc = le_arr.transform([arrival])[0]

                input_df = pd.DataFrame([[dep_enc, arr_enc, month, duration]], 
                                        columns=['Gare_depart_enc', 'Gare_arrivee_enc', 'Mois', 'Durée moyenne du trajet'])

                prediction = model.predict(input_df)[0]
                
                st.markdown("---")
                st.subheader("Résultat de l'analyse")
                
                if prediction < 5:
                    st.success(f"Retard estimé : {prediction:.1f} minutes.")
                    st.progress(10)
                    st.info("Tout devrait bien se passer. Profitez de votre voyage !")
                elif prediction < 15:
                    st.warning(f"Retard estimé : {prediction:.1f} minutes.")
                    st.progress(50)
                    st.info("Risque modéré. Prévoyez une petite marge pour votre correspondance.")
                else:
                    st.error(f"Retard estimé : {prediction:.1f} minutes.")
                    st.progress(90)
                    st.info("Risque élevé. L'historique de ce trajet est capricieux, prenez vos précautions.")

            except Exception as e:
                st.error(f"Erreur de calcul de l'IA : {e}")