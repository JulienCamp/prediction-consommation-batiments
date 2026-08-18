"""Démonstrateur Streamlit de prédiction de consommation énergétique."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from inference import build_scenario, load_artifacts, predict_scenario
from modeling import KBTU_TO_MWH


st.set_page_config(page_title="Consommation des bâtiments", page_icon=":material/energy_savings_leaf:", layout="wide")

SQFT_PER_SQM = 10.7639104167


@st.cache_resource
def cached_artifacts():
    return load_artifacts()


def mwh(kbtu: float) -> float:
    return kbtu * KBTU_TO_MWH


def sqm(square_feet: float) -> float:
    return square_feet / SQFT_PER_SQM


def energy_label(kbtu: float) -> str:
    value = mwh(kbtu)
    if value >= 1_000:
        return f"{value / 1_000:,.2f} GWh".replace(",", " ")
    return f"{value:,.0f} MWh".replace(",", " ")


def comparison_chart(values: list[tuple[str, float, str]]) -> alt.Chart:
    chart_data = pd.DataFrame(
        {
            "Estimation": [label for label, _, _ in values],
            "Consommation (MWh)": [mwh(value) for _, value, _ in values],
            "Nature": [nature for _, _, nature in values],
        }
    )
    return (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusEnd=5)
        .encode(
            x=alt.X("Consommation (MWh):Q", title="Consommation annuelle (MWh)"),
            y=alt.Y("Estimation:N", sort=[label for label, _, _ in values], title=None),
            color=alt.Color(
                "Nature:N",
                scale=alt.Scale(
                    domain=["Modèle", "Mesure 2016", "Repère statistique"],
                    range=["#13795B", "#577590", "#B76812"],
                ),
                legend=None,
            ),
            tooltip=[alt.Tooltip("Estimation:N"), alt.Tooltip("Consommation (MWh):Q", format=",.0f")],
        )
        .properties(height=260)
    )


without_score_model, with_score_model, metadata = cached_artifacts()
profiles = metadata["profiles"]
demo_metrics = metadata["metrics"]["with_energy_star"]

st.title("Anticiper la consommation d'un bâtiment")
st.write(
    "Explorez l'estimation d'un modèle entraîné sur les données publiques 2016 de Seattle, "
    "puis mesurez l'apport de l'ENERGY STAR Score."
)
st.badge("Démonstrateur éducatif", icon=":material/science:", color="blue")

mode = st.segmented_control(
    "Choisir un parcours",
    ["Explorer des exemples", "Simuler un bâtiment"],
    default="Explorer des exemples",
    key="app_mode",
    width="stretch",
)

with st.expander("Comprendre l'ENERGY STAR Score", icon=":material/energy_savings_leaf:"):
    st.markdown(
        """
        L'**ENERGY STAR Score** est un indice de 1 à 100 développé par l'Agence américaine de protection
        de l'environnement. Il situe la performance énergétique d'un bâtiment par rapport à des bâtiments
        comparables, en tenant compte notamment de son activité et de ses conditions d'exploitation.
        Un score élevé correspond généralement à une meilleure performance relative.

        Dans cette application, deux modèles ont été entraînés séparément : l'un reçoit ce score comme
        information supplémentaire, l'autre utilise uniquement les caractéristiques structurelles et d'usage.
        Leur écart indique donc **ce que cette information change dans l'estimation du modèle**. Il ne mesure
        pas les économies qui seraient causées par une amélioration du score.

        [En savoir plus sur ENERGY STAR](https://www.energystar.gov/about?s=mega)
        """
    )

with st.sidebar:
    st.subheader("Repères du modèle")
    st.metric("Bâtiments de test", f"{metadata['training']['test_rows']}")
    st.metric("R² en unités réelles", f"{demo_metrics['r2']:.2f}".replace(".", ","))
    st.metric(
        "Erreur absolue médiane",
        f"{demo_metrics['median_absolute_error_kbtu'] / 1_000_000:.2f} M kBtu".replace(".", ","),
    )
    st.caption("Variante compacte validée en août 2026 sur des données tenues à l'écart de l'entraînement.")

if mode == "Explorer des exemples":
    st.subheader("Explorer des bâtiments observés", anchor=False)
    st.caption(
        "Ces trois bâtiments appartiennent au jeu de test. Leurs caractéristiques sont fixes, "
        "ce qui permet de comparer honnêtement les estimations à la mesure de 2016."
    )
    profile = st.selectbox(
        "Bâtiment exemple",
        profiles,
        format_func=lambda item: f"{item['label']} — {item['description']}",
        key="example_profile",
    )
    features = profile["features"]
    predictions = predict_scenario(
        without_score_model, with_score_model, build_scenario(profile, {})
    )
    with_kbtu = predictions["with_energy_star_kbtu"]
    without_kbtu = predictions["without_energy_star_kbtu"]
    observed_kbtu = profile["observed_kbtu"]

    details_col, output_col = st.columns([0.8, 1.2], gap="large")
    with details_col:
        with st.container(border=True):
            st.subheader("Caractéristiques", anchor=False)
            st.metric("Usage principal", features["PrimaryPropertyType"])
            st.metric("Type de bâtiment", features["BuildingType"])
            with st.container(horizontal=True):
                surface_sqm = round(sqm(features["PropertyGFABuilding(s)"]))
                st.metric("Surface", f"{surface_sqm:,} m²".replace(",", " "), border=True)
                st.metric("Âge", f"{int(features['PropertyAge'])} ans", border=True)
                st.metric("Étages", f"{max(1, int(features['NumberofFloors']))}", border=True)
            with st.container(horizontal=True):
                st.metric("ENERGY STAR Score", f"{int(round(features['ENERGYSTARScore']))}", border=True)
                st.metric("Part du gaz", f"{features['NaturalGasProportion']:.0%}", border=True)
                st.metric("Part de vapeur", f"{features['SteamProportion']:.0%}", border=True)
            if profile["source_energy_star_missing"]:
                st.caption("Le score ENERGY STAR était absent et a été remplacé par la médiane d'entraînement.")

    with output_col:
        with st.container(border=True):
            st.subheader("Prédiction et mesure", anchor=False)
            with st.container(horizontal=True):
                st.metric("Avec ENERGY STAR Score", energy_label(with_kbtu), border=True)
                st.metric("Sans ENERGY STAR Score", energy_label(without_kbtu), border=True)
                st.metric("Consommation observée", energy_label(observed_kbtu), border=True)
            st.altair_chart(
                comparison_chart(
                    [
                        ("Avec score", with_kbtu, "Modèle"),
                        ("Sans score", without_kbtu, "Modèle"),
                        ("Mesure observée", observed_kbtu, "Mesure 2016"),
                    ]
                )
            )
            error = (with_kbtu - observed_kbtu) / observed_kbtu if observed_kbtu else 0
            st.caption(f"Pour cet exemple, l'estimation avec score s'écarte de la mesure de {error:+.1%}.")
        with st.container(border=True):
            st.subheader("Comment lire cet exemple ?", anchor=False)
            st.write(
                "L'écart montre qu'une prédiction individuelle peut être imparfaite, même lorsque "
                "le modèle améliore les résultats en moyenne sur le jeu de test."
            )

else:
    st.subheader("Simuler un bâtiment", anchor=False)
    st.caption(
        "Construisez un scénario fictif et observez la sensibilité du modèle. "
        "Aucune consommation réelle n'est associée à ce scénario."
    )
    base_profile = profiles[1]
    features = base_profile["features"]
    input_col, output_col = st.columns([0.9, 1.1], gap="large")
    with input_col:
        with st.container(border=True):
            st.subheader("Décrire le scénario", anchor=False)
            st.caption("Les valeurs initiales constituent un exemple médian modifiable.")
            with st.form("custom_scenario"):
                surface_sqm = st.number_input(
                    "Surface du bâtiment (m²)", 100, 185_000,
                    round(sqm(features["PropertyGFABuilding(s)"])), 500, key="custom_surface"
                )
                property_type = st.selectbox(
                    "Usage principal", metadata["categories"]["property_types"],
                    index=metadata["categories"]["property_types"].index(features["PrimaryPropertyType"]),
                    key="custom_property",
                )
                building_type = st.selectbox(
                    "Type de bâtiment", metadata["categories"]["building_types"],
                    index=metadata["categories"]["building_types"].index(features["BuildingType"]),
                    key="custom_building",
                )
                age = st.slider("Âge du bâtiment (années)", 0, 130, int(features["PropertyAge"]), key="custom_age")
                floors = st.number_input(
                    "Nombre d'étages", 1, 100, max(1, int(features["NumberofFloors"])), key="custom_floors"
                )
                energy_star = st.slider(
                    "ENERGY STAR Score", 1, 100, int(round(features["ENERGYSTARScore"])),
                    help="Indice comparatif de 1 à 100 : un score élevé correspond généralement à une meilleure performance énergétique relative.",
                    key="custom_score",
                )
                with st.expander("Paramètres énergétiques", icon=":material/tune:"):
                    natural_gas = st.slider(
                        "Part du gaz naturel", 0.0, 1.0, float(features["NaturalGasProportion"]),
                        0.01, format="%.2f", key="custom_gas"
                    )
                    steam = st.slider(
                        "Part de la vapeur", 0.0, 1.0, float(features["SteamProportion"]),
                        0.01, format="%.2f", key="custom_steam"
                    )
                st.form_submit_button("Estimer la consommation", type="primary", icon=":material/bolt:")

    updates = {
        # Le modèle reste alimenté dans l'unité d'origine du jeu de données de Seattle.
        "PropertyGFABuilding(s)": surface_sqm * SQFT_PER_SQM,
        "PrimaryPropertyType": property_type,
        "BuildingType": building_type,
        "PropertyAge": age,
        "NumberofFloors": floors,
        "ENERGYSTARScore": energy_star,
        "NaturalGasProportion": natural_gas,
        "SteamProportion": steam,
    }
    predictions = predict_scenario(
        without_score_model, with_score_model, build_scenario(base_profile, updates)
    )
    with_kbtu = predictions["with_energy_star_kbtu"]
    without_kbtu = predictions["without_energy_star_kbtu"]
    median_kbtu = metadata["benchmarks_kbtu"]["median"]
    with output_col:
        with st.container(border=True):
            st.subheader("Estimation du scénario", anchor=False)
            with st.container(horizontal=True):
                st.metric("Avec ENERGY STAR Score", energy_label(with_kbtu), border=True)
                st.metric("Sans ENERGY STAR Score", energy_label(without_kbtu), border=True)
                st.metric("Médiane du jeu de test", energy_label(median_kbtu), border=True)
            st.altair_chart(
                comparison_chart(
                    [
                        ("Avec score", with_kbtu, "Modèle"),
                        ("Sans score", without_kbtu, "Modèle"),
                        ("Médiane du parc", median_kbtu, "Repère statistique"),
                    ]
                )
            )
            difference = (with_kbtu - without_kbtu) / without_kbtu if without_kbtu else 0
            st.caption(f"Dans ce scénario, le score modifie l'estimation de {difference:+.1%}.")
        with st.container(border=True):
            st.subheader("Comment lire cette simulation ?", anchor=False)
            st.write(
                "La médiane donne un repère de taille, mais ne constitue pas la consommation attendue "
                "d'un bâtiment comparable. Modifier une variable montre une sensibilité du modèle, "
                "pas un effet causal garanti."
            )
            st.warning(
                "Cette estimation sert à explorer un ordre de grandeur. Elle ne remplace ni un relevé "
                "énergétique ni un audit de terrain.", icon=":material/warning:"
            )

with st.expander("Méthode et limites", icon=":material/info:"):
    st.markdown(
        """
        - **Données :** 1 535 bâtiments non résidentiels de Seattle, année 2016.
        - **Modèle :** Random Forest compact de 300 arbres, entraîné sur `log1p(consommation)`, puis reconverti en kBtu et MWh.
        - **Validation :** 461 bâtiments tenus à l'écart ; les métriques affichées dans la barre latérale sont celles de cette variante.
        - **Modèle d'étude :** la revalidation complète à 1 500 arbres reste documentée séparément dans le portfolio.
        - **Portée :** démonstration éducative issue d'un projet OpenClassrooms, sans usage réglementaire ou opérationnel.
        """
    )

st.caption("Projet OpenClassrooms réalisé en 2023 · Revalidation et démonstrateur portfolio en 2026")
