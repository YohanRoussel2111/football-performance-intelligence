"""Science & Méthodologie — la justification scientifique derrière chaque
indicateur et chaque choix de modélisation, avec les références clés."""
from __future__ import annotations

import streamlit as st

from app import app_core, ui


def render():
    ui.header("Science & Méthodologie", "la preuve derrière chaque indicateur")

    st.markdown(
        "> **Question centrale :** comment transformer des données de performance "
        "longitudinales et complexes en une intelligence *explicable et actionnable* "
        "qui aide le staff à décider de la disponibilité et de la performance des joueurs ?")
    st.caption("Chaque variable et chaque choix de modélisation ci-dessous s'appuie "
               "sur la littérature en sciences du sport et en apprentissage automatique. "
               "Les données de démonstration sont synthétiques, mais la méthode est réelle.")

    st.divider()

    # ------------------------------------------------------------------ Charge
    st.subheader("1 · Charge externe & interne")
    st.markdown(
        "La **charge externe** (GPS : distance, course haute vitesse, accélérations, "
        "player load) est le travail mécanique réalisé ; la **charge interne** "
        "(sRPE, fréquence cardiaque) est la réponse physiologique à ce travail "
        "(Impellizzeri et al., 2019). On les suit ensemble car un même travail externe "
        "peut coûter très différemment selon l'état du joueur.")
    ui.science_box(
        "<b>sRPE (session-RPE)</b> = RPE × durée : une mesure simple et validée de la "
        "charge interne d'entraînement (Foster et al., 2001).")
    ui.science_box(
        "<b>ACWR</b> (ratio charge aiguë:chronique) compare la charge récente (~7 j) à "
        "la charge de fond (~28 j). Une charge aiguë bien supérieure à la charge "
        "chronique traduit un pic de fatigue relatif (Gabbett, 2016). ⚠️ La forme "
        "couplée par moyennes mobiles est bruitée et critiquée : on l'expose ici avec "
        "prudence, jamais comme une règle, et on montre aussi les fenêtres aiguë et "
        "chronique brutes (Impellizzeri et al., 2020 ; Lolli et al., 2019).")
    ui.science_box(
        "<b>Monotonie</b> (moyenne ÷ écart-type de la charge quotidienne) et "
        "<b>contrainte / strain</b> (charge hebdo × monotonie) : une charge élevée ET "
        "monotone est plus contraignante qu'une charge variée (Foster, 1998).")

    # -------------------------------------------------------------- Bien-être
    st.subheader("2 · Bien-être auto-déclaré")
    st.markdown(
        "Les questionnaires courts (fatigue, courbatures, sommeil, stress, humeur) "
        "sont des marqueurs sensibles, peu coûteux et non invasifs de la réponse à la "
        "charge et de la récupération. Ils réagissent souvent aux variations aiguës de "
        "charge mieux que des marqueurs objectifs isolés (Saw et al., 2016 ; Hooper & "
        "Mackinnon, 1995).")
    ui.science_box(
        "On lit le bien-être en <b>écart z par rapport à la référence propre au "
        "joueur</b> : une note de 4/7 ne signifie pas la même chose chez un joueur "
        "habituellement à 6 que chez un joueur habituellement à 3.")

    # -------------------------------------------------------------- Neuromusc.
    st.subheader("3 · Statut neuromusculaire (CMJ)")
    st.markdown(
        "Le **saut avec contre-mouvement (CMJ)** est un test de terrain validé de la "
        "fatigue neuromusculaire. Une baisse marquée de la hauteur (ou des variables de "
        "temps/force) par rapport à la référence individuelle signale une récupération "
        "incomplète (Claudino et al., 2017 ; Gathercole et al., 2015).")

    # -------------------------------------------------------------- Congestion
    st.subheader("4 · Exposition & congestion des matchs")
    st.markdown(
        "Les minutes jouées, le nombre de matchs sur 14/28 jours et le délai depuis le "
        "dernier match capturent l'**exposition compétitive**. La congestion des "
        "calendriers est associée à une récupération incomplète et à une disponibilité "
        "réduite (Carling et al., 2016 ; Julian et al., 2021).")

    st.divider()

    # ------------------------------------------------------------------ ML
    st.subheader("5 · Cadre d'apprentissage automatique")
    st.markdown(
        "**Cible (sans fuite) :** un jour où le joueur est *disponible*, entrera-t-il "
        f"dans un état de disponibilité réduite dans les "
        f"{app_core.config.PREDICTION_HORIZON_DAYS} prochains jours ? On se restreint "
        "aux jours où le joueur est disponible pour apprendre une *transition*, pas la "
        "persistance d'un épisode en cours.")
    ui.science_box(
        "Les épisodes étant <b>rares</b>, on privilégie la <b>PR-AUC</b> (précision-"
        "rappel) plutôt que l'exactitude, trompeuse en cas de déséquilibre (Saito & "
        "Rehmsmeier, 2015). Le déséquilibre est géré au niveau de l'estimateur "
        "(class_weight / scale_pos_weight), sans rééchantillonnage, pour préserver la "
        "calibration.")
    ui.science_box(
        "<b>Validation temporelle</b> : on découpe strictement par le temps (début de "
        "saison → entraînement, milieu → validation, fin → test). Un split aléatoire "
        "ferait fuiter du futur et des mesures répétées d'un même joueur, gonflant le "
        "score. On prédit toujours le futur à partir du passé (Luo et al., 2016).")
    ui.science_box(
        "<b>Calibration</b> (Platt scaling) : une décision opérationnelle s'appuie sur "
        "le *niveau* d'une probabilité, pas seulement son classement. On mesure la "
        "qualité par le <b>score de Brier</b> et la courbe de fiabilité (Van Calster "
        "et al., 2019).")
    ui.science_box(
        "<b>Explicabilité (SHAP)</b> : chaque prédiction est décomposée en "
        "contributions signées par variable (Lundberg & Lee, 2017), puis traduite en "
        "langage praticien. On passe de « le modèle dit ÉLEVÉ » à « voici pourquoi ».")

    st.divider()

    # -------------------------------------------------------------- Leakage
    st.subheader("6 · Éviter la fuite de données")
    st.markdown(
        "Toute variable connue seulement au moment de l'événement (ou après) — les "
        "libellés de disponibilité eux-mêmes, ou toute variable qui en dérive — est "
        "sur **liste noire** et exclue des variables du modèle. Un test unitaire échoue "
        "si une variable interdite est utilisée. Toutes les variables glissantes/"
        "références sont **causales** (calculées uniquement à partir du passé jusqu'au "
        "jour t). C'est la discipline qui rend l'évaluation honnête et le modèle "
        "déployable.")

    # -------------------------------------------------------------- Limites
    st.subheader("7 · Limites & éthique")
    st.markdown(
        "- **Données synthétiques** : aucune donnée médicale réelle, aucune blessure "
        "cliniquement validée. Les chiffres décrivent le simulateur, pas le monde réel.\n"
        "- **Aide à la décision, pas diagnostic** : le modèle recommande une *revue par "
        "le staff*, jamais « ne pas s'entraîner » ni « ce joueur va se blesser ».\n"
        "- **Corrélation ≠ causalité** : une variable qui augmente le score n'est pas "
        "pour autant une *cause*.\n"
        "- **Validation externe requise** : une utilisation réelle exigerait une "
        "étude prospective, multi-clubs, et une validation clinique/performance.")

    with st.expander("📚 Références clés"):
        st.markdown(
            "- Foster C. (1998). *Monitoring training in athletes with reference to "
            "overtraining syndrome.* Med Sci Sports Exerc.\n"
            "- Foster C. et al. (2001). *A new approach to monitoring exercise "
            "training (session-RPE).* J Strength Cond Res.\n"
            "- Hooper S.L., Mackinnon L.T. (1995). *Monitoring overtraining in "
            "athletes.* Sports Med.\n"
            "- Gabbett T.J. (2016). *The training—injury prevention paradox.* Br J "
            "Sports Med.\n"
            "- Gathercole R. et al. (2015). *Alterations in neuromuscular function "
            "after a soccer match (CMJ).* Int J Sports Physiol Perform.\n"
            "- Saw A.E. et al. (2016). *Monitoring the athlete training response: "
            "subjective self-reported measures.* Br J Sports Med.\n"
            "- Carling C. et al. (2016). *Match running performance during fixture "
            "congestion in elite soccer.* Sports Med.\n"
            "- Claudino J.G. et al. (2017). *The countermovement jump to monitor "
            "neuromuscular status.* J Sci Med Sport.\n"
            "- Impellizzeri F.M. et al. (2019). *Internal and external training load.* "
            "Int J Sports Physiol Perform.\n"
            "- Impellizzeri F.M. et al. (2020). *Acute:chronic workload ratio: "
            "conceptual and methodological issues.* Int J Sports Physiol Perform.\n"
            "- Lolli L. et al. (2019). *Mathematical coupling of the ACWR.* Br J "
            "Sports Med.\n"
            "- Julian R. et al. (2021). *Fixture congestion in football.* Sports Med.\n"
            "- Saito T., Rehmsmeier M. (2015). *The precision-recall plot is more "
            "informative than ROC for imbalanced data.* PLoS ONE.\n"
            "- Luo W. et al. (2016). *Guidelines for developing and reporting ML "
            "predictive models in biomedical research.* J Med Internet Res.\n"
            "- Van Calster B. et al. (2019). *Calibration: the Achilles heel of "
            "predictive analytics.* BMC Medicine.\n"
            "- Lundberg S., Lee S. (2017). *A unified approach to interpreting model "
            "predictions (SHAP).* NeurIPS.")
        st.caption("Références indicatives destinées à situer la démarche ; à vérifier "
                   "et compléter avant tout usage réel.")

    ui.disclaimer()
