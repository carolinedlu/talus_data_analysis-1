import base64

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from weasyprint import CSS, HTML
from weasyprint.formatting_structure.boxes import InlineReplacedBox
from talus_data_analysis.plot import *
from talus_data_analysis.load import read_df_from_s3
from talus_data_analysis.save import streamlit_report_to_pdf

st.set_page_config(
    page_title="Talus Data Analysis",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()

ENCYCLOPEDIA_BUCKET = "talus-data-pipeline-encyclopedia-bucket"

def clean_protein_name(name):
    return name.split("|")[-1].replace("_HUMAN", "")

@st.cache(allow_output_mutation=True, hash_funcs={pd.DataFrame: lambda _: None})
def get_data(key):
    df = read_df_from_s3(bucket=ENCYCLOPEDIA_BUCKET, key=key, inputformat="parquet")
    # df = pd.read_csv("data/MLLTX/msstats_groupcompare.csv")
    df = df[
        [
            "Protein",
            "Label",
            "log2FC",
            "SE",
            "Tvalue",
            "DF",
            "pvalue",
            "adj.pvalue",
            "issue",
        ]
    ]
    df["ProteinName"] = df["Protein"].apply(clean_protein_name)
    return df


@st.cache(allow_output_mutation=True, hash_funcs={pd.DataFrame: lambda _: None})
def get_volcano_fig(df, filter_labels, subset_label):
    filtered_df = df[df["Label"] == subset_label]
    return volcano(
        df=filtered_df,
        log_fold_change_col="log2FC",
        pvalue_col="adj.pvalue",
        pvalue_threshold=(0.05, 0.05),
        log_fold_change_threshold=(2 ** 2, 2 ** 2),
        label_column_name="ProteinName",
        filter_labels=filter_labels,
        title="Volcano Plot mapping the log2 fold change and the log10 adjusted p-value for each Protein",
        dim=(900, 750),
        sign_line=True,
    )


@st.cache(allow_output_mutation=True, hash_funcs={pd.DataFrame: lambda _: None})
def get_heatmap_fig(df, start_idx, value_column, filter_labels, sort_ascending=None):
    return heatmap(
        df=df,
        log_fold_change_col="log2FC",
        pvalue_col="adj.pvalue",
        start_idx=start_idx,
        log_fold_change_threshold=(2**2, 2**2),
        pvalue_threshold=(0.05, 0.05),
        sort_ascending=sort_ascending,
        x_label_column_name="Label",
        y_label_column_name="ProteinName",
        value_column_name=value_column,
        filter_labels=filter_labels,
        title="Heatmap showing the log2 fold change for a selection of Proteins for each Sample Comparison",
        dim=(750, 1000),
    )


@st.cache(allow_output_mutation=True, hash_funcs={pd.DataFrame: lambda _: None})
def get_comparison_fig(df):
    return comparison(
        df=df,
        log_fold_change_col="log2FC",
        pvalue_col="adj.pvalue",
        pvalue_threshold=0.05,
        label_column_name="Label",
        title="Comparison plot mapping the log2 fold change for a protein across all Sample Comparisons",
        xaxis_title="Comparison",
        yaxis_title="log2 fold change",
        dim=(750, 750),
    )


######################################
# Data Analysis
######################################

TITLE_TEXT = "Data Analysis" 
st.title(TITLE_TEXT)

st.sidebar.header("Options")

dataset_choices = ["210308_MLLtx", "210308_MM1S", "210511_GryderLab", "210511_DalalLab", "210525_BJ", "210525_Gryder"]
dataset = st.sidebar.selectbox("Dataset", options=dataset_choices)

show_volcano = st.sidebar.checkbox("Volcano Plot", key="vol")
show_heatmap = st.sidebar.checkbox("Heatmap Plot", key="hm")
show_comparison = st.sidebar.checkbox("Comparison Plot", key="cmp")

df = get_data(key=f"wide/{dataset}/msstats_groupcompare.parquet")
all_proteins = df["ProteinName"].unique().tolist()
all_labels = df["Label"].unique().tolist()

st.sidebar.subheader("Filter by Protein")
uploaded_files = st.sidebar.file_uploader("Choose a file (.csv)", accept_multiple_files=True)
filter_proteins = set()
for i, uploaded_file in enumerate(uploaded_files):
    filter_df = pd.read_csv(uploaded_file)
    st.sidebar.write(uploaded_file.name)
    protein_col = st.sidebar.selectbox("Select Protein Column", options=list(filter_df.columns), key=str(i))
    filter_proteins.update(list(map(clean_protein_name, filter_df[protein_col].tolist())))

default_filter_proteins = [protein for protein in filter_proteins if protein in set(all_proteins)]
proteins = st.sidebar.multiselect("Proteins", options=all_proteins, default=default_filter_proteins)

figures = []
descriptions = []

if show_volcano:
    st.header("Volcano Plot")
    st.sidebar.header("Volcano Plot")
    labels = st.sidebar.multiselect("Select Drug Interaction", options=df["Label"].unique(), default=None)
    for label in labels:
        fig_v = get_volcano_fig(df=df, filter_labels=proteins, subset_label=label)
        st.write(fig_v)
        figures.append(fig_v)

        volcano_placeholder = f"A volcano Plot mapping the log2 fold change (x-axis) against the log10 adjusted p-value (y-axis) for each Protein for the drug interaction: {label}. Values that were inf or -inf were limited to a maximum value and can be seen in the upper right and left corner respectively."
        volcano_description = st.text_area(
            "Description", key="volcano", value=volcano_placeholder
        )
        descriptions.append(volcano_description)

        presence_absence = volcano_list_presence_absence(
            df=df[df["Label"] == label], log_fold_change_col="log2FC"
        )
        # descriptions.append(f'{volcano_description} Presence: {", ".join([v for v in presence_absence["Presence"].values if v != ""])}. Absence: {", ".join([v for v in presence_absence["Absence"].values if v != ""])}')
        with st.beta_expander("Show Presence/Absence Proteins"):
            st.dataframe(presence_absence)
        st.text("")

if show_heatmap:
    NUM_PROTEINS = 50
    st.header("Heatmap Plot")
    st.sidebar.header("Heatmap Plot")
    start = st.sidebar.slider(
        "Select start of range (50 proteins)",
        min_value=0,
        max_value=len(df["ProteinName"].unique()) - NUM_PROTEINS,
    )
    labels_hm = st.sidebar.multiselect(
        "Filter by Drug Interaction", options=all_labels, default=None
    )
    value_column = st.sidebar.selectbox("Select Value Column", ["log2FC", "adj.pvalue"])
    hm_sort_ascending = st.sidebar.selectbox(
        "Sort Ascending", options=[False, True, None]
    )
    fig_hm = get_heatmap_fig(
        df,
        start_idx=start,
        value_column=value_column,
        filter_labels=[("Label", labels_hm), ("ProteinName", proteins)],
        sort_ascending=hm_sort_ascending,
    )
    st.write(fig_hm)
    figures.append(fig_hm)
    heatmap_placerholder = "A heatmap showing the log2 fold change for a selection of Proteins for each Sample Comparison. The values below a threshold have been filtered out."
    heatmap_description = st.text_area(
        "Description", key="volcano", value=heatmap_placerholder
    )
    descriptions.append(heatmap_description)

if show_comparison:
    st.header("Comparison Plot")
    if proteins:
        protein_df = df[df["ProteinName"] == proteins[0]]
        fig_cmp = get_comparison_fig(df=protein_df)
        st.write(fig_cmp)
        figures.append(fig_cmp)
        comp_placerholder = "A comparison plot mapping the log2 fold change for a protein across all sample comparisons. The function used to calculate the error margin is scipy.stats's Percent Point Function (ppf)."
        comp_description = st.text_area(
            "Description", key="volcano", value=comp_placerholder
        )
        descriptions.append(comp_description)

    else:
        st.info("Please select a Protein to compare.")

######################################
# Export
######################################

if st.button("Export to PDF"):
    with st.spinner(text="Loading"):
        streamlit_report_to_pdf(title=TITLE_TEXT, 
                                dataset_name=dataset, 
                                figures=figures, 
                                descriptions=descriptions, 
                                write_html=False)