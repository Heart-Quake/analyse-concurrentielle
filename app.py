import streamlit as st
import pandas as pd
import re
from io import BytesIO
import plotly.express as px

st.set_page_config(
    page_title="Analyse Concurrentielle",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour améliorer l'apparence
st.markdown("""
    <style>
    /* Hiérarchie visuelle */
    h1, h2, h3, h4, h5, h6 {
        color: #2BAF9C;  /* Vert du logo */
    }
    
    /* Style spécifique pour les titres de filtres */
    .filter-title {
        color: #2BAF9C;
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    /* Style pour les titres de sections */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        color: #2BAF9C;
        font-size: 1rem;
        font-weight: 500;
    }
    
    /* Composants interactifs - Boutons et téléchargements */
    .stButton > button, .stDownloadButton > button, div[data-testid="stSidebarNav"] button {
        background-color: #2BAF9C;  /* Vert du logo */
        color: white;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border-radius: 0.5rem;
        border: none;
        transition: all 0.2s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover, div[data-testid="stSidebarNav"] button:hover {
        background-color: #249889;  /* Version plus foncée du vert */
        transform: translateY(-1px);
    }
    
    /* Tableaux et métriques */
    .dataframe {
        font-size: 0.9rem;
        border-radius: 0.5rem;
        border: 1px solid #E5E8E8;
    }
    .metric-container {
        background-color: #F8F9F9;
        padding: 1.25rem;
        border-radius: 0.5rem;
        border: 1px solid #E5E8E8;
    }
    
    /* Filtres */
    .stSlider {
        padding: 1rem 0;
    }
    .stMultiSelect {
        margin-bottom: 1rem;
    }

    /* Titres des sections */
    .section-header {
        color: #249889;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

def process_csv_files(uploaded_files, client_name):
    """Traitement des fichiers CSV avec validation améliorée"""
    try:
        list_dfs = []
        processed_domains = set()
        
        for uploaded_file in uploaded_files:
            try:
                # Validation du nom de fichier
                if not uploaded_file.name.endswith('.csv'):
                    st.warning(f"Le fichier {uploaded_file.name} n'est pas un fichier CSV")
                    continue

                # Extraction et validation du domaine
                domain_match = re.search(r"([^\.]+)\.", uploaded_file.name)
                if not domain_match:
                    st.warning(f"Impossible d'extraire le domaine du fichier {uploaded_file.name}")
                    continue
                
                domain = domain_match.group(1)
                if domain == client_name:
                    is_client = True
                else:
                    is_client = False

                # Détection du format et lecture
                try:
                    if "organic-keywords" in uploaded_file.name:
                        df_temp = pd.read_csv(uploaded_file, encoding='utf-16le', sep='\t')
                        column_mapping = {
                            'Keyword': 'Keyword',
                            'Current position': 'Position',
                            'Volume': 'Search Volume',
                            'KD': 'Keyword Difficulty',
                            'CPC': 'CPC',
                            'Current URL': 'URL'
                        }
                    elif "organic.Positions" in uploaded_file.name:
                        df_temp = pd.read_csv(uploaded_file, encoding='utf-8', sep=',')
                        column_mapping = {
                            'Keyword': 'Keyword',
                            'Position': 'Position',
                            'Search Volume': 'Search Volume',
                            'Keyword Difficulty': 'Keyword Difficulty',
                            'CPC': 'CPC',
                            'URL': 'URL'
                        }
                    else:
                        st.warning(f"Format non reconnu pour {uploaded_file.name}")
                        continue
                except UnicodeDecodeError:
                    st.error(f"Erreur d'encodage pour {uploaded_file.name}")
                    continue

                # Validation des colonnes
                missing_columns = [col for col in column_mapping.keys() if col not in df_temp.columns]
                if missing_columns:
                    st.error(f"Colonnes manquantes dans {uploaded_file.name}: {', '.join(missing_columns)}")
                    continue

                # Nettoyage et standardisation
                df_temp = df_temp.rename(columns=column_mapping)
                df_temp = df_temp[list(column_mapping.values())]
                
                # Conversion et nettoyage des données
                df_temp['Position'] = pd.to_numeric(df_temp['Position'], errors='coerce')
                df_temp['Search Volume'] = pd.to_numeric(df_temp['Search Volume'].astype(str).replace(',', ''), errors='coerce')
                df_temp['Keyword Difficulty'] = pd.to_numeric(df_temp['Keyword Difficulty'], errors='coerce')
                df_temp['CPC'] = pd.to_numeric(df_temp['CPC'].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce')
                
                # Nettoyage des valeurs nulles
                df_temp = df_temp.dropna(subset=['Keyword'])
                df_temp = df_temp.fillna({
                    'Position': 0,
                    'Search Volume': 0,
                    'Keyword Difficulty': 0,
                    'CPC': 0
                })

                # Ajout des métadonnées
                df_temp['Domain'] = domain
                df_temp['Is_Client'] = is_client
                processed_domains.add(domain)
                
                list_dfs.append(df_temp)
                st.success(f"Fichier traité avec succès : {uploaded_file.name}")
                
            except Exception as e:
                st.error(f"Erreur lors du traitement de {uploaded_file.name}: {str(e)}")
                continue

        if not list_dfs:
            st.error("Aucun fichier n'a pu être traité correctement")
            return None

        # Validation finale
        if client_name not in processed_domains:
            st.error(f"Aucun fichier trouvé pour le client {client_name}")
            return None

        # Concaténation et validation finale
        df_combined = pd.concat(list_dfs, ignore_index=True)
        if len(df_combined) == 0:
            st.error("Aucune donnée valide trouvée dans les fichiers")
            return None

        st.info(f"Nombre total de domaines traités : {len(processed_domains)}")
        return df_combined

    except Exception as e:
        st.error(f"Erreur lors du traitement global : {str(e)}")
        return None

def define_strategy(position):
    """Définit la stratégie en fonction de la position"""
    if pd.isnull(position) or position == 0:
        return "Non positionné"
    elif position == 1:
        return 'Sauvegarde'
    elif 2 <= position <= 5:
        return 'Quick Win'
    elif 6 <= position <= 10:
        return 'Opportunité'
    elif 11 <= position <= 20:
        return 'Potentiel'
    else:
        return 'Conquête'

def initialize_session_state():
    """Initialise les variables de session"""
    if 'df_final' not in st.session_state:
        st.session_state.df_final = None
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False

def process_and_store_data(uploaded_files, client_name, nombre_sites, top_position):
    """Traitement des données avec validation améliorée"""
    try:
        with st.spinner("Analyse en cours..."):
            # Traitement initial des fichiers
            df_combined = process_csv_files(uploaded_files, client_name)
            if df_combined is None:
                return False

            # Création du pivot avec gestion correcte des positions
            pivot_pos = df_combined.pivot_table(
                values='Position',
                index=['Keyword', 'Search Volume', 'Keyword Difficulty', 'CPC'],
                columns='Domain',
                aggfunc='min'
            )

            # Validation des données du client
            if client_name not in pivot_pos.columns:
                st.error(f"Données du client {client_name} non trouvées")
                return False

            # Correction du filtre pour le nombre minimum de sites
            # Compter le nombre de sites positionnés dans le top X pour chaque mot-clé
            sites_in_top = (pivot_pos <= top_position).sum(axis=1)
            
            # Filtrer les mots-clés qui ont au moins N sites positionnés
            mask = sites_in_top >= nombre_sites
            filtered_keywords = pivot_pos[mask].index
            
            if len(filtered_keywords) == 0:
                st.warning(f"Aucun mot-clé n'a au moins {nombre_sites} sites positionnés dans le top {top_position}")
                return False

            # Création du DataFrame final
            df_filtered = df_combined[
                df_combined.set_index(['Keyword', 'Search Volume', 'Keyword Difficulty', 'CPC'])
                .index.isin(filtered_keywords)
            ].reset_index(drop=True)

            # Création des pivots finaux
            pivot_pos = create_position_pivot(df_filtered, client_name)
            pivot_url = create_url_pivot(df_filtered, client_name)

            # Création du DataFrame final
            df_final = create_final_dataframe(pivot_pos, pivot_url, client_name)

            # Stockage des résultats
            st.session_state.df_final = df_final
            st.session_state.analysis_done = True
            
            # Afficher des informations sur le filtrage
            st.info(f"""
            Résultats du filtrage :
            - Nombre total de mots-clés : {len(pivot_pos)}
            - Mots-clés retenus : {len(filtered_keywords)}
            - Critères : au moins {nombre_sites} sites dans le top {top_position}
            """)
            
            return True

    except Exception as e:
        st.error(f"Erreur lors du traitement : {str(e)}")
        return False

def display_filtered_results(filtered_df, client_name):
    """Affiche les résultats filtrés"""
    with st.expander("🔍 Filtres avancés", expanded=True):
        # Filtres par stratégie
        st.markdown('<p class="filter-title">Filtrer par stratégie</p>', unsafe_allow_html=True)
        strategies = filtered_df['Stratégie'].unique().tolist()
        selected_strategies = st.multiselect(
            "",
            options=strategies,
            default=strategies,
            key='strategy_filter',
            help="""
            Sauvegarde : Position 1
            Quick Win : Positions 2-5
            Opportunité : Positions 6-10
            Potentiel : Positions 11-20
            Conquête : Positions > 20
            Non positionné : Absent du top 100
            """
        )

        # Métriques principales
        st.markdown('<p class="filter-title">Métriques principales</p>', unsafe_allow_html=True)
        metric_cols = st.columns(3)
        
        # Volume de recherche
        with metric_cols[0]:
            vol_min = int(filtered_df['Search Volume'].min())
            vol_max = int(filtered_df['Search Volume'].max())
            volume_range = st.slider(
                "Volume de recherche",
                min_value=vol_min,
                max_value=vol_max,
                value=(vol_min, vol_max),
                key='volume_slider',
                help="Nombre mensuel moyen de recherches pour ce mot-clé"
            )

        # Position
        with metric_cols[1]:
            position_min = 0
            position_max = int(filtered_df[f'{client_name} (Position)'].max())
            position_range = st.slider(
                "Position du client",
                min_value=position_min,
                max_value=position_max,
                value=(position_min, min(20, position_max)),
                help="Position actuelle du site client dans les résultats Google"
            )

        # Difficulté
        with metric_cols[2]:
            kd_range = st.slider(
                "Difficulté",
                min_value=int(filtered_df['Keyword Difficulty'].min()),
                max_value=int(filtered_df['Keyword Difficulty'].max()),
                value=(0, 100),
                key='kd_slider',
                help="Score de difficulté (0-100) pour se positionner sur ce mot-clé"
            )

        # Recherche et tri
        st.markdown('<p class="filter-title">Recherche et tri</p>', unsafe_allow_html=True)
        cols = st.columns([2, 1.5, 1])
        
        # Recherche
        keyword_search = cols[0].text_input(
            "",
            placeholder="Rechercher des mots-clés",
            help="Filtrer les résultats contenant ce texte dans le mot-clé",
            key='keyword_search'
        )

        # Tri
        sort_options = {
            'Volume de recherche': 'Search Volume',
            'Difficulté': 'Keyword Difficulty',
            'Position': f'{client_name} (Position)',
            'CPC': 'CPC'
        }
        sort_by = cols[1].selectbox(
            "",
            options=list(sort_options.keys()),
            format_func=lambda x: f"Trier par : {x}",
            help="Choisir le critère de tri des résultats"
        )

        # Ordre
        sort_order = cols[2].radio(
            "",
            options=['Décroissant', 'Croissant'],
            horizontal=True,
            help="Ordre de tri des résultats"
        )

        # Intention de recherche
        st.markdown('<p class="filter-title">Intention de recherche</p>', unsafe_allow_html=True)
        intentions = filtered_df['Intention'].unique().tolist()
        selected_intentions = st.multiselect(
            "",
            options=intentions,
            default=intentions,
            key='intent_filter',
            help="""
            Informationnelle : Recherche d'information (comment, pourquoi...)
            Transactionnelle : Intention d'achat (prix, acheter...)
            Navigationnelle : Recherche de site/page spécifique
            Commerciale : Comparaison/évaluation de produits
            Autre : Intention non déterminée
            """
        )

    # Application des filtres
    filtered_df = filtered_df[
        (filtered_df['Stratégie'].isin(selected_strategies)) &
        (filtered_df['Intention'].isin(selected_intentions)) &
        (filtered_df['Search Volume'].between(volume_range[0], volume_range[1])) &
        (filtered_df[f'{client_name} (Position)'].fillna(0).between(position_range[0], position_range[1])) &
        (filtered_df['Keyword Difficulty'].between(kd_range[0], kd_range[1]))
    ]

    # Application du filtre textuel
    if keyword_search:
        filtered_df = filtered_df[
            filtered_df['Keyword'].str.contains(keyword_search, case=False, na=False)
        ]
    
    # Application du tri
    sort_column = sort_options[sort_by]
    filtered_df = filtered_df.sort_values(
        by=sort_column,
        ascending=(sort_order == 'Croissant')
    )

    # Métriques principales avec icônes
    st.markdown('<p class="subheader">📈 Synthèse des mots-clés</p>', unsafe_allow_html=True)
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        st.metric("🎯 Total mots-clés", f"{len(filtered_df):,}")
    with metric_col2:
        st.metric("📊 Volume total", f"{int(filtered_df['Search Volume'].sum()):,}")
    with metric_col3:
        st.metric("📈 KD moyen", f"{round(filtered_df['Keyword Difficulty'].mean(), 1)}")
    with metric_col4:
        st.metric("💰 CPC moyen", f"${round(filtered_df['CPC'].mean(), 2)}")

    # Création des onglets
    tab1, tab2, tab3 = st.tabs(["📊 Résultats filtrés", "📈 Répartition par stratégie", "🔍 Visualisations"])
    
    with tab1:
        st.dataframe(filtered_df, use_container_width=True, height=400)
        export_data(filtered_df, client_name)

    with tab2:
        # Définir l'ordre des stratégies
        strategy_order = ['Sauvegarde', 'Quick Win', 'Opportunité', 'Potentiel', 'Conquête', 'Non positionné']
        
        # Calcul des statistiques avec un ordre fixe
        stats_df = filtered_df.groupby('Stratégie').agg({
            'Keyword': 'count',
            'Search Volume': 'sum',
            'Keyword Difficulty': 'mean',
            'CPC': 'mean'
        }).round(2)
        
        # Réorganiser selon l'ordre défini
        stats_df = stats_df.reindex(strategy_order)
        
        # Renommer les colonnes
        stats_df.columns = ['Nombre de mots-clés', 'Volume total', 'KD moyen', 'CPC moyen']
        
        # Formater les valeurs numériques
        stats_df['Nombre de mots-clés'] = stats_df['Nombre de mots-clés'].fillna(0).astype(int)
        stats_df['Volume total'] = stats_df['Volume total'].fillna(0).astype(int)
        stats_df['KD moyen'] = stats_df['KD moyen'].round(1)
        stats_df['CPC moyen'] = stats_df['CPC moyen'].round(2)
        
        st.dataframe(stats_df, use_container_width=True, height=230)

    with tab3:
        col1, col2, col3 = st.columns(3)
        
        # Configuration commune pour tous les graphiques
        graph_title_style = {
            'font': {'size': 16, 'family': 'Arial'},
            'y': 0.95  # Position verticale du titre
        }
        graph_layout = {
            'title_font': graph_title_style['font'],
            'showlegend': True,
            'paper_bgcolor': 'white',
            'plot_bgcolor': 'white',
            'margin': dict(t=50, b=30, l=30, r=30)
        }
        
        with col1:
            # Distribution des volumes par position
            fig_volume = create_position_volume_histogram(filtered_df, client_name)
            fig_volume.update_layout(
                title=dict(
                    text='Distribution du volume de recherche par position',
                    **graph_title_style
                ),
                **graph_layout
            )
            st.plotly_chart(fig_volume, use_container_width=True)
        
        with col2:
            # Répartition des stratégies
            fig_strategies = px.pie(
                filtered_df,
                names='Stratégie',
                title='Répartition des stratégies',
                category_orders={'Stratégie': strategy_order}
            )
            fig_strategies.update_layout(
                title=dict(
                    text='Répartition des stratégies',
                    **graph_title_style
                ),
                **graph_layout
            )
            st.plotly_chart(fig_strategies, use_container_width=True)
        
        with col3:
            # Répartition des intentions
            fig_intentions = px.pie(
                filtered_df,
                names='Intention',
                title='Répartition des intentions de recherche'
            )
            fig_intentions.update_layout(
                title=dict(
                    text='Répartition des intentions de recherche',
                    **graph_title_style
                ),
                **graph_layout
            )
            st.plotly_chart(fig_intentions, use_container_width=True)

def create_position_volume_histogram(filtered_df, client_name):
    """Crée un histogramme de distribution des volumes par position"""
    # Création des catégories de position
    def categorize_position(pos):
        if pd.isnull(pos) or pos == 0:
            return 'Non positionné'
        elif pos <= 3:
            return 'Top 3'
        elif pos <= 10:
            return 'Top 4-10'
        elif pos <= 20:
            return 'Top 11-20'
        else:
            return 'Top 21-100'

    # Ajout de la catégorie de position
    df_viz = filtered_df.copy()
    df_viz['Position_Category'] = df_viz[f'{client_name} (Position)'].apply(categorize_position)
    
    # Calcul de la somme des volumes par catégorie
    position_volume = df_viz.groupby('Position_Category')['Search Volume'].sum().reset_index()
    
    # Définir l'ordre personnalisé des catégories avec "Non positionné" à la fin
    category_order = ['Top 3', 'Top 4-10', 'Top 11-20', 'Top 21-100', 'Non positionné']
    position_volume['Position_Category'] = pd.Categorical(
        position_volume['Position_Category'],
        categories=category_order,
        ordered=True
    )
    
    # Création du graphique
    fig_volume = px.bar(
        position_volume.sort_values('Position_Category'),
        x='Position_Category',
        y='Search Volume',
        title='Distribution du volume de recherche par position',
        labels={
            'Position_Category': 'Position',
            'Search Volume': 'Volume de recherche'
        }
    )
    
    # Personnalisation améliorée du graphique
    fig_volume.update_layout(
        title_font_size=20,
        title_font_family="Arial",
        plot_bgcolor='white',
        paper_bgcolor='white',
        bargap=0.2,
        margin=dict(t=50, b=50, l=50, r=25),
        xaxis=dict(
            title_font_size=14,
            tickfont_size=12,
            gridcolor='#E5E8E8'
        ),
        yaxis=dict(
            title_font_size=14,
            tickfont_size=12,
            gridcolor='#E5E8E8'
        )
    )
    
    # Ajout de couleurs personnalisées
    fig_volume.update_traces(
        marker_color=['#2ECC71', '#3498DB', '#F1C40F', '#E67E22', '#E74C3C']
    )
    
    return fig_volume

def export_to_excel(filtered_df, client_name):
    """Export Excel avec mise en forme"""
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        filtered_df.to_excel(writer, sheet_name='Analyse', index=False)
        workbook = writer.book
        worksheet = writer.sheets['Analyse']
        
        # Formats
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#2BAF9C',  # Vert du logo
            'color': 'white'
        })
        
        # Appliquer les formats
        for col_num, value in enumerate(filtered_df.columns.values):
            worksheet.write(0, col_num, value, header_format)
            worksheet.set_column(col_num, col_num, 15)
    
    return buffer

def add_help_tooltips():
    """Ajoute des infobulles d'aide détaillées"""
    st.sidebar.markdown("""
    # 📖 Guide d'utilisation

    ## 1. Import des données
    ### Formats acceptés :
    - **Ahrefs** : `domain-organic-keywords.csv`
       - Export depuis : Organic Keywords > Export
       - Encodage : UTF-16
    - **Semrush** : `domain-organic.Positions.csv`
       - Export depuis : Organic Research > Positions
       - Encodage : UTF-8

    ## 2. Configuration
    - **Client** : Sélectionnez votre domaine
    - **Nombre minimum de sites** : Filtrer les mots-clés présents sur X sites concurrents
    - **Position maximum** : Limite de position pour l'analyse (ex: top 20)

    ## 3. Stratégies SEO
    - **🏆 Sauvegarde** (Pos. 1) : Maintenir le positionnement
    - **⚡ Quick Win** (Pos. 2-5) : Opportunités rapides
    - **📈 Opportunité** (Pos. 6-10) : Potentiel à court terme
    - **🎯 Potentiel** (Pos. 11-20) : Potentiel à moyen terme
    - **🚀 Conquête** (Pos. > 20) : Objectifs long terme

    ## 4. Intentions de recherche
    - **ℹ️ Informationnelle** : Recherche d'information
    - **💰 Transactionnelle** : Intention d'achat
    - **🔍 Navigationnelle** : Recherche de site/marque
    - **🛒 Commerciale** : Comparaison/évaluation
    """)

def add_tooltips_to_filters():
    """Ajoute des tooltips aux éléments de filtrage"""
    tooltips = {
        "volume": "Nombre mensuel moyen de recherches pour ce mot-clé",
        "kd": """Score de difficulté (0-100) :
        - 0-20 : Facile
        - 21-40 : Modéré
        - 41-60 : Difficile
        - 61-80 : Très difficile
        - 81-100 : Extrêmement difficile""",
        "position": "Position actuelle du site client dans les résultats Google",
        "cpc": "Coût par clic moyen en publicité Google Ads",
        "concurrence": "Nombre de sites positionnés sur ce mot-clé"
    }
    return tooltips

def add_contextual_help():
    """Ajoute des explications contextuelles dans l'interface"""
    with st.expander("ℹ️ Comment utiliser cet outil ?", expanded=False):
        st.markdown("""
        ### Processus en 4 étapes :

        1. **Préparation des données**
           - Exportez les données de vos outils SEO
           - Assurez-vous d'avoir les fichiers pour chaque concurrent
           - Nommez les fichiers avec le domaine (ex: monsite.csv)

        2. **Import et configuration**
           - Importez tous vos fichiers en une fois
           - Sélectionnez votre domaine client
           - Ajustez les paramètres d'analyse selon vos besoins

        3. **Analyse des résultats**
           - Utilisez les filtres pour affiner votre analyse
           - Examinez les différentes visualisations
           - Identifiez les opportunités prioritaires

        4. **Export et action**
           - Exportez les résultats filtrés
           - Utilisez les données pour votre stratégie SEO
           - Suivez l'évolution des positions

        ### Conseils d'utilisation :
        - Commencez par les "Quick Wins" pour des résultats rapides
        - Analysez l'intention de recherche pour prioriser vos actions
        - Utilisez les filtres de volume pour identifier les opportunités à fort potentiel
        """)

def add_metric_explanations():
    """Ajoute des explications pour chaque métrique"""
    with st.expander("📊 Comprendre les métriques", expanded=False):
        st.markdown("""
        ### Métriques principales

        #### 🎯 Stratégie
        - **Sauvegarde** : Mots-clés en position 1 - Focus sur la défense
        - **Quick Win** : Positions 2-5 - Potentiel de gain rapide
        - **Opportunité** : Positions 6-10 - Progression possible
        - **Potentiel** : Positions 11-20 - Travail à moyen terme
        - **Conquête** : Positions > 20 - Objectif long terme

        #### 📈 Métriques SEO
        - **Volume** : Nombre moyen de recherches mensuelles
        - **KD** : Score de difficulté (0-100)
            - 0-20 : Facile
            - 21-40 : Modéré
            - 41-60 : Difficile
            - 61-80 : Très difficile
            - 81-100 : Extrêmement difficile
        
        #### 🎯 Intention de recherche
        - **Informationnelle** : Recherche d'information
        - **Transactionnelle** : Intention d'achat
        - **Navigationnelle** : Recherche de site/marque
        - **Commerciale** : Comparaison/évaluation
        """)

def export_data(filtered_df, client_name):
    """Export des données au format CSV"""
    st.download_button(
        "📥 Export CSV",
        filtered_df.to_csv(index=False),
        f"Analyse_Concurrentielle_{client_name}.csv",
        mime="text/csv",
        help="Télécharger les résultats au format CSV",
        type="primary"
    )

def create_advanced_visualizations(filtered_df, client_name):
    # Distribution des volumes par difficulté
    fig_volume_kd = px.scatter(
        filtered_df,
        x='Keyword Difficulty',
        y='Search Volume',
        color='Stratégie',
        size='Search Volume',
        hover_data=['Keyword'],
        title='Volume vs Difficulté par stratégie'
    )
    
    # Évolution des positions
    fig_positions = px.box(
        filtered_df,
        x='Stratégie',
        y=f'{client_name} (Position)',
        title='Distribution des positions par stratégie'
    )
    
    return fig_volume_kd, fig_positions

def add_advanced_filters(df_final):
    with st.expander("Filtres avancés supplémentaires"):
        col1, col2 = st.columns(2)
        with col1:
            # Filtre par CPC
            cpc_range = st.slider(
                "CPC ($)",
                min_value=float(df_final['CPC'].min()),
                max_value=float(df_final['CPC'].max()),
                value=(0.0, float(df_final['CPC'].max()))
            )
        with col2:
            # Filtre par longueur de mot-clé
            df_final['Keyword_Length'] = df_final['Keyword'].str.split().str.len()
            length_range = st.slider(
                "Nombre de mots",
                min_value=int(df_final['Keyword_Length'].min()),
                max_value=int(df_final['Keyword_Length'].max()),
                value=(1, int(df_final['Keyword_Length'].max()))
            )

def save_load_settings():
    if 'settings' not in st.session_state:
        st.session_state.settings = {}
    
    with st.sidebar:
        with st.expander("Paramètres de session"):
            if st.button("Sauvegarder les paramètres"):
                st.session_state.settings = {
                    'client_name': client_name,
                    'nombre_sites': nombre_sites,
                    'top_position': top_position,
                    'filters': current_filters
                }
                st.success("Paramètres sauvegardés!")
            
            if st.button("Charger les paramètres"):
                if st.session_state.settings:
                    load_saved_settings(st.session_state.settings)

def handle_file_processing(uploaded_files):
    with st.spinner("Traitement des fichiers..."):
        try:
            progress_bar = st.progress(0)
            for i, file in enumerate(uploaded_files):
                # Traitement du fichier
                progress_bar.progress((i + 1) / len(uploaded_files))
                
        except Exception as e:
            st.error(f"Erreur : {str(e)}")
            st.info("""
            🔍 Solutions possibles :
            1. Vérifiez le format des fichiers
            2. Assurez-vous que toutes les colonnes requises sont présentes
            3. Vérifiez l'encodage des fichiers
            """)

def create_position_pivot(df_filtered, client_name):
    """Crée le pivot des positions"""
    pivot_pos = df_filtered.pivot_table(
        values='Position',
        index=['Keyword', 'Search Volume', 'Keyword Difficulty', 'CPC'],
        columns='Domain',
        aggfunc='first'
    )
    
    # Renommage des colonnes
    pivot_pos.columns = [f"{col} (Position)" for col in pivot_pos.columns]
    
    # Réorganisation pour mettre le client en premier
    pos_cols = pivot_pos.columns.tolist()
    client_pos_col = f"{client_name} (Position)"
    if client_pos_col in pos_cols:
        pos_cols.remove(client_pos_col)
        pos_cols = [client_pos_col] + sorted(pos_cols)
        pivot_pos = pivot_pos[pos_cols]
    
    return pivot_pos

def create_url_pivot(df_filtered, client_name):
    """Crée le pivot des URLs"""
    pivot_url = df_filtered.pivot_table(
        values='URL',
        index=['Keyword', 'Search Volume', 'Keyword Difficulty', 'CPC'],
        columns='Domain',
        aggfunc='first'
    )
    
    # Renommage des colonnes
    pivot_url.columns = [f"{col} (URL)" for col in pivot_url.columns]
    
    # Réorganisation pour mettre le client en premier
    url_cols = pivot_url.columns.tolist()
    client_url_col = f"{client_name} (URL)"
    if client_url_col in url_cols:
        url_cols.remove(client_url_col)
        url_cols = [client_url_col] + sorted(url_cols)
        pivot_url = pivot_url[url_cols]
    
    return pivot_url

def create_final_dataframe(pivot_pos, pivot_url, client_name):
    """Crée le DataFrame final avec toutes les colonnes organisées"""
    # Concaténation des pivots
    df_final = pd.concat([pivot_pos, pivot_url], axis=1).reset_index()
    
    # Ajout de la stratégie
    df_final['Stratégie'] = df_final.apply(
        lambda x: define_strategy(x[f'{client_name} (Position)']), 
        axis=1
    )
    
    # Ajout de l'intention de recherche
    df_final['Intention'] = df_final['Keyword'].apply(lambda x: classify_intent(x))
    
    # Calcul du nombre de sites positionnés (Concurrence)
    position_columns = [col for col in df_final.columns if col.endswith('(Position)')]
    df_final['Concurrence'] = df_final[position_columns].apply(
        lambda x: (x > 0).sum(),
        axis=1
    )
    
    # Ajout de la colonne de sélection
    df_final['Sélection'] = ''
    
    # Organisation des colonnes
    base_cols = ['Keyword', 'Sélection', 'Stratégie', 'Intention', 'Concurrence', 'Search Volume', 'Keyword Difficulty', 'CPC']
    pos_cols = [col for col in pivot_pos.columns if col.endswith('(Position)')]
    url_cols = [col for col in pivot_url.columns if col.endswith('(URL)')]
    
    # Organisation finale
    df_final = df_final[base_cols + pos_cols + url_cols]
    
    return df_final

def classify_intent(keyword, language="Français"):
    """Classifie l'intention de recherche d'un mot-clé"""
    patterns = {
        "Français": {
            'Informationnelle': [r"""\b(comment|pourquoi|quel(?:le|s)?|exemple|fabrication|fabriquer|quand|où|qui|quoi|impacts?|étapes?|fonctionnement|fonctionne|c'est quoi|est[-\s]?il|problèmes?|faire|pouvoir|est[-\s]?ce que|manières?|consignes?|guide|doit|combien|histoire|astuces?|définitions?|peut|veut|significations?|aide|avantages?|inconvénients?|dangers?|risques?|inconvénients?|changements?|faut[-\s]il|change|pour|en cas de|information(s)?)\b"""],
            'Transactionnelle': [r"""\b(abonnement|moins chers?|pas chers?|achet(?:er|é)|achats?|gratuit(?:e)?|command(?:er|é)|prix|tarif(?:s)?|coûts?|en ligne|réserver|boutique|magasin|offre(s)?|solde(s)?|remise(s)?|promot(?:ion|ionnelle?)|coupon(s)?|réduction(s)?)\b"""],
            'Navigationnelle': [r"""\b(mon compte|compte client|site|page|espace client|téléphone|service client|accueil|login|connexion|inscription|profil(s)?|dashboard|plateforme|url|web|officiel)\b"""],
            'Commerciale': [r"""\b(meilleur|top|compar(?:er|aison)|avis|reviews?|tests?|compar|simul(?:ation)?|choisir|quelle marque|quel modèle|recommand(?:ation|é)|guides? d'achat)\b"""]
        }
    }

    for intent, exprs in patterns[language].items():
        for expr in exprs:
            if re.search(expr, keyword, re.IGNORECASE):
                return intent
    return 'Autre'

def extract_domains_from_files(uploaded_files):
    """Extrait les domaines des fichiers téléchargés"""
    domains = set()
    for file in uploaded_files:
        domain_match = re.search(r"([^\.]+)\.", file.name)
        if domain_match:
            domains.add(domain_match.group(1))
    return sorted(list(domains))

def main():
    initialize_session_state()
    
    # En-tête avec description
    st.title("Analyse Concurrentielle")
    
    # Guide d'utilisation en dropdown dans la zone principale
    with st.expander("📖 Guide d'utilisation", expanded=False):
        st.markdown("""
        ## 1. Import des données
        ### Formats acceptés :
        - **Ahrefs** : `domain-organic-keywords.csv`
           - Export depuis : Organic Keywords > Export
           - Encodage : UTF-16
        - **Semrush** : `domain-organic.Positions.csv`
           - Export depuis : Organic Research > Positions
           - Encodage : UTF-8

        ## 2. Configuration
        - **Client** : Sélectionnez votre domaine
        - **Nombre minimum de sites** : Filtrer les mots-clés présents sur X sites concurrents
        - **Position maximum** : Limite de position pour l'analyse (ex: top 20)

        ## 3. Stratégies SEO
        - **🏆 Sauvegarde** (Pos. 1) : Maintenir le positionnement
        - **⚡ Quick Win** (Pos. 2-5) : Opportunités rapides
        - **📈 Opportunité** (Pos. 6-10) : Potentiel à court terme
        - **🎯 Potentiel** (Pos. 11-20) : Potentiel à moyen terme
        - **🚀 Conquête** (Pos. > 20) : Objectifs long terme

        ## 4. Intentions de recherche
        - **ℹ️ Informationnelle** : Recherche d'information
        - **💰 Transactionnelle** : Intention d'achat
        - **🔍 Navigationnelle** : Recherche de site/marque
        - **🛒 Commerciale** : Comparaison/évaluation

        **💡 Conseil** : Commencez par les "Quick Wins" (positions 2-5) pour des résultats rapides
        """)
    
    # Configuration dans la sidebar
    with st.sidebar:
        # Ajout du logo
        st.image("DR SEO Header.svg", use_column_width=True)
        
        st.header("Configuration")
        
        # Étape 1 : Téléchargement des fichiers
        uploaded_files = st.file_uploader(
            "Importer les fichiers CSV",
            accept_multiple_files=True,
            type=['csv'],
            help="Formats acceptés : Ahrefs (organic-keywords.csv) ou Semrush (organic.Positions.csv)"
        )

        if uploaded_files:
            # Étape 2 : Sélection du client
            domains = extract_domains_from_files(uploaded_files)
            client_name = st.selectbox(
                "Sélectionner le client",
                options=domains,
                help="Sélectionnez le domaine correspondant au client"
            )
            
            # Paramètres avancés
            st.subheader("Paramètres d'analyse")
            nombre_sites = st.number_input("Nombre minimum de sites", min_value=1, value=1)
            top_position = st.number_input("Position maximum", min_value=1, value=20)

            # Bouton d'action
            st.markdown("---")
            if client_name:
                if st.button("Lancer l'analyse", type="primary"):
                    if process_and_store_data(uploaded_files, client_name, nombre_sites, top_position):
                        st.success("Analyse terminée avec succès!")

    # Affichage des résultats si disponibles
    if st.session_state.analysis_done and st.session_state.df_final is not None:
        display_filtered_results(st.session_state.df_final, client_name)

if __name__ == "__main__":
    main() 