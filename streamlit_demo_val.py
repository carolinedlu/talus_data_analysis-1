import numpy as np
import pandas as pd
import streamlit as st

from dotenv import load_dotenv
from sklearn.decomposition import PCA

from talus_data_analysis.load import get_s3_file_sizes
from talus_data_analysis.load import read_df_from_s3
from talus_data_analysis.plot import box
from talus_data_analysis.plot import clustergram
from talus_data_analysis.plot import correlation
from talus_data_analysis.plot import histogram
from talus_data_analysis.plot import pca
from talus_data_analysis.plot import scatter_matrix
from talus_data_analysis.plot import venn
from talus_data_analysis.save import streamlit_report_to_pdf


st.set_page_config(
    page_title="Talus Data Validation",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()

width = 600
height = 600

ENCYCLOPEDIA_BUCKET = "talus-data-pipeline-encyclopedia-bucket"
LANDING_BUCKET = "talus-data-pipeline-landing-bucket"
COLLECTIONS_BUCKET = "protein-collections"


@st.cache(allow_output_mutation=True, hash_funcs={pd.DataFrame: lambda _: None})
def get_val_data(key):
    df = read_df_from_s3(bucket=ENCYCLOPEDIA_BUCKET, key=key, inputformat="parquet")
    # df = pd.read_csv("data/MM1S/elib2msstats_output.csv")
    return df


@st.cache(allow_output_mutation=True, hash_funcs={pd.DataFrame: lambda _: None})
def get_protein_data(key):
    df = read_df_from_s3(bucket=ENCYCLOPEDIA_BUCKET, key=key, inputformat="txt")
    # df = pd.read_table("data/MM1S/RESULTS-quant.elib.proteins.txt")
    return df


@st.cache(allow_output_mutation=True, hash_funcs={pd.DataFrame: lambda _: None})
def get_peptide_data(key):
    df = read_df_from_s3(bucket=ENCYCLOPEDIA_BUCKET, key=key, inputformat="txt")
    # df = pd.read_table("data/MM1S/RESULTS-quant.elib.peptides.txt")
    df = df.set_index(["Peptide"])
    df = df.drop(["Protein", "numFragments"], axis=1)
    return df


@st.cache(allow_output_mutation=True, hash_funcs={pd.DataFrame: lambda _: None})
def get_nuclear_protein_data(key):
    df = read_df_from_s3(bucket=COLLECTIONS_BUCKET, key=key, inputformat="csv")
    # df = pd.read_csv(key)
    return df


@st.cache(allow_output_mutation=True, hash_funcs={pd.DataFrame: lambda _: None})
def get_file_sizes(bucket, dataset):
    return get_s3_file_sizes(
        bucket=bucket, keys=[f"narrow/{dataset}", f"wide/{dataset}"], file_type=".raw"
    )


@st.cache(allow_output_mutation=True, hash_funcs={pd.DataFrame: lambda _: None})
def get_venn_diagram(sets, labels, title, colors=None):
    return venn(sets=sets, labels=labels, title=title, colors=colors, dim=(800, 500))


@st.cache(allow_output_mutation=True, hash_funcs={pd.DataFrame: lambda _: None})
def get_correlation_fig(df, replicates, filter_outliers, color):
    return correlation(
        x=df[df_val["BioReplicate"] == replicates[0]]["Intensity"].values,
        y=df[df_val["BioReplicate"] == replicates[1]]["Intensity"].values,
        filter_outliers=filter_outliers,
        title="Correlation Plot plotting all intensity values for two given samples",
        xaxis_title="replicate 1",
        yaxis_title="replicate 2",
        trendline_color=color,
        dim=(750, 750),
    )


@st.cache(allow_output_mutation=True, hash_funcs={pd.DataFrame: lambda _: None})
def get_scatter_matrix_fig(df, filter_outliers, color, opacity=None):
    return scatter_matrix(
        df,
        x_label_column_name="Condition",
        y_label_column_name="ProteinName",
        value_column_name="Intensity",
        filter_outliers=filter_outliers,
        log_scaling=True,
        title="Scatter Matrix Plot of Peptide Intensities for each Sample",
        color=color,
        opacity=opacity,
        dim=(900, 900),
    )


@st.cache(allow_output_mutation=True, hash_funcs={pd.DataFrame: lambda _: None})
def get_box_fig(
    df,
    x_label_column_name,
    y_label_column_name,
    log_scaling,
    filter_outliers,
    title,
    color,
):
    return box(
        df=df,
        x_label_column_name=x_label_column_name,
        y_label_column_name=y_label_column_name,
        log_scaling=log_scaling,
        filter_outliers=filter_outliers,
        show_points=False,
        title=title,
        xaxis_title="Sample",
        yaxis_title="log2_intensity",
        color=color,
        dim=(750, 900),
    )


@st.cache(allow_output_mutation=True, hash_funcs={pd.DataFrame: lambda _: None})
def get_histogram_fig(df, color):
    return histogram(
        df,
        x_label_column_name="NumPeptides",
        xaxis_title="# of Peptides",
        yaxis_title="Number of Proteins",
        title="Histogram Plot mapping the Distribution of the Number of Peptides detected for each Protein",
        color=color,
        value_cutoff=30,
        dim=(900, 750),
    )


@st.cache(allow_output_mutation=True, hash_funcs={pd.DataFrame: lambda _: None})
def get_pca_fig(pca_components, labels, color, explained_variance_ratio):
    return pca(
        pca_components=pca_components,
        labels=labels,
        dim=(750, 750),
        title="PCA Plot mapping the Principal Components of the Peptide Intensities for each Sample",
        xaxis_title=f"Principal Component 1 ({round(explained_variance_ratio[0]*100, 2)}%)",
        yaxis_title=f"Principal Component 2 ({round(explained_variance_ratio[1]*100, 2)}%)",
        color=color,
    )


@st.cache(allow_output_mutation=True, hash_funcs={pd.DataFrame: lambda _: None})
def get_clustergram_fig(df, column_labels):
    return clustergram(
        df=df,
        column_labels=column_labels,
        title="Clustergram Plot mapping the log10 Peptide Intensities for each Sample",
        dim=(750, 1000),
    )


TITLE_TEXT = "Data Validation"
st.title(TITLE_TEXT)

st.sidebar.header("Options")

dataset_choices = [
    "210308_MLLtx",
    "210308_MM1S",
    "210511_GryderLab",
    "210511_DalalLab",
    "210525_BJ",
    "210525_Gryder",
]
dataset = st.sidebar.selectbox("Dataset", options=dataset_choices)

primary_color = st.sidebar.color_picker("Primary Color", "#001425")
secondary_color = st.sidebar.color_picker("Secondary Color", "#308AAD")

show_file_sizes = st.sidebar.checkbox("File Sizes", key="file_sizes")
show_venn = st.sidebar.checkbox("Venn Diagram", key="venn")
show_correlation = st.sidebar.checkbox("Correlation Plot", key="corr")
show_scatter_matrix = st.sidebar.checkbox("Scatter Matrix Plot", key="sm")
show_box = st.sidebar.checkbox("Box Plot", key="box")
show_hist = st.sidebar.checkbox("Histogram Plot", key="hist")
show_pca = st.sidebar.checkbox("PCA Plot", key="pca")
show_clustergram = st.sidebar.checkbox("Clustergram Plot", key="cluster")


######################################
# Validation
######################################

df_val = get_val_data(key=f"wide/{dataset}/peptide_proteins_results.parquet")
df_val_processed = get_val_data(
    key=f"wide/{dataset}/peptide_proteins_normalized.parquet"
)
df_nuclear_proteins = get_nuclear_protein_data(key="nuclear_proteins.csv")
file_to_condition = (
    df_val[["Run", "Condition"]]
    .drop_duplicates()
    .set_index("Run")
    .to_dict()["Condition"]
)

all_drugs = df_val[df_val["Condition"].notna()]["Condition"].unique().tolist()

figures = []
descriptions = []

if show_file_sizes:
    st.header(".raw file sizes")
    file_sizes_df = get_file_sizes(bucket=LANDING_BUCKET, dataset=dataset)
    st.dataframe(file_sizes_df)

if show_venn:
    st.header("Venn Diagram")
    st.sidebar.header("Venn Diagram")
    st.sidebar.subheader("Custom Protein List")
    uploaded_files = st.sidebar.file_uploader(
        "Choose a file (.csv)", accept_multiple_files=True
    )
    custom_proteins = set()
    for i, uploaded_file in enumerate(uploaded_files):
        custom_df = pd.read_csv(uploaded_file)
        st.sidebar.write(uploaded_file.name)
        protein_col = st.sidebar.selectbox(
            "Select Protein Column", options=list(custom_df.columns), key=str(i)
        )
        custom_proteins.update(list(custom_df[protein_col].tolist()))

    if not custom_proteins:
        protein_col = df_nuclear_proteins.columns[-1]
        custom_proteins = df_nuclear_proteins[protein_col].unique()
    measured_proteins = set(
        df_val["ProteinName"]
        .str.upper()
        .apply(lambda x: x.split("|")[-1].replace("_HUMAN", ""))
    )
    fig_venn = get_venn_diagram(
        sets=[set(custom_proteins), set(measured_proteins)],
        labels=[protein_col, "Measured Proteins"],
        colors=[primary_color, secondary_color],
        title="A Venn Diagram showing the overlap between a list of nuclear proteins and the measured proteins",
    )
    st.write(fig_venn)
    figures.append(fig_venn)

    venn_placeholder = "A Venn diagram showing the overlap between a list of nuclear proteins and the proteins that were measured during these sample runs."
    venn_description = st.text_area("Description", key="venn", value=venn_placeholder)
    descriptions.append(venn_description)

if show_correlation:
    st.header("Correlation Plot")
    st.sidebar.header("Correlation Plot")
    drug = st.sidebar.selectbox("Filter by Drug", options=all_drugs)
    corr_filter_outliers = st.sidebar.checkbox("Filter outliers", key="1")

    if drug:
        st.write(drug)
        replicates = (
            df_val[df_val["Condition"] == drug]["BioReplicate"].unique().tolist()
        )

        if len(replicates) < 2:
            st.write(
                f"Not enough replicates: {len(replicates)}. Needs at least 2 to compare."
            )
        else:
            fig_corr = get_correlation_fig(
                df=df_val,
                replicates=replicates,
                filter_outliers=corr_filter_outliers,
                color=primary_color,
            )
            st.write(fig_corr)
            figures.append(fig_corr)

        corr_description = st.text_area(
            "Description",
            key="corr",
            value="A correlation plot showing all intensity values for two given samples. It can be used to compare the intensity values for the same samples during different runs. Optionally outliers can be filtered out.",
        )
        descriptions.append(corr_description)

if show_scatter_matrix:
    st.header("Scatter Matrix Plot")
    st.sidebar.header("Scatter Matrix Plot")
    sm_filter_outliers = st.sidebar.checkbox("Filter outliers", key="2")
    sm_opacity = st.sidebar.slider(
        "Point Opacity", min_value=0, max_value=100, value=50
    )

    fig_sm = get_scatter_matrix_fig(
        df=df_val,
        filter_outliers=sm_filter_outliers,
        color=primary_color,
        opacity=sm_opacity / 100,
    )
    st.write(fig_sm)
    figures.append(fig_sm)

    scatter_placeholder = "A scatter matrix plot containing the log10 protein intensities for each sample. The diagonal displays each sample mapped against itself which is why it is a straight line. Points falling far from x=y represent outliers. The farther a pair of samples (a point) falls from x=y, the more uncorrelated it is. In order to fit outliers the axes are sometimes adjusted and are not necessarily all the same."
    scatter_description = st.text_area(
        "Description", key="scatter", value=scatter_placeholder
    )
    descriptions.append(scatter_description)


if show_box:
    st.header("Box Plot")

    with st.spinner(text="Loading"):
        col1, col2 = st.beta_columns(2)

        # Raw
        st.sidebar.header("Box Plot (raw)")
        box_filter_outliers = st.sidebar.checkbox("Filter outliers", key="3")
        fig_box = get_box_fig(
            df=df_val,
            x_label_column_name="Condition",
            y_label_column_name="Intensity",
            log_scaling=True,
            filter_outliers=box_filter_outliers,
            title="Box Plot of Raw Peptides Intensities for each Sample",
            color=primary_color,
        )
        col1.write(fig_box)
        figures.append(fig_box)

        box_placeholder = "A box plot showing the log2 peptide intensities for each sample/replicate. The outliers are filtered out and the ends of the box represent the lower (25th) and upper (75th) quartiles, while the median (second quartile) is marked by a line inside the box. If the distribution of one sample deviates from the others, that sample is an outlier."
        box_description = col1.text_area(
            "Description", key="box", value=box_placeholder
        )
        descriptions.append(box_description)

        # Normalized
        st.sidebar.header("Box Plot (normalized)")
        box_norm_filter_outliers = st.sidebar.checkbox("Filter outliers", key="4")

        fig_box_norm = get_box_fig(
            df=df_val_processed,
            x_label_column_name="GROUP_ORIGINAL",
            y_label_column_name="ABUNDANCE",
            log_scaling=False,
            filter_outliers=box_norm_filter_outliers,
            title="Box Plot of Normalized Peptides Intensities for each Sample",
            color=primary_color,
        )
        col2.write(fig_box_norm)
        figures.append(fig_box_norm)

        box_norm_placeholder = "A box plot showing the same data as above but the intensities have been log2 transformed and normalized."
        box_norm_description = col2.text_area(
            "Description", key="box_norm", value=box_norm_placeholder
        )
        descriptions.append(box_norm_description)

######################################
# Protein Data
######################################

if show_hist:
    df_proteins = get_protein_data(
        key=f"wide/{dataset}/RESULTS-quant.elib.proteins.txt"
    )

    st.header("Histogram Plot")
    fig_hist = get_histogram_fig(df=df_proteins, color=primary_color)
    st.write(fig_hist)
    figures.append(fig_hist)

    hist_placeholder = "A histogram plotting the distribution of the number of peptides detected for each protein. It uses the data from the final report and therefore represents the data across all runs. The last bar to the right represents a catch-all and includes everything above this value. Ideally we should have more than two peptides for each protein but the more the better. The more peptides we have, the more confident we are in a detection. Having only one peptide could be due to randomness."
    hist_description = st.text_area("Description", key="hist", value=hist_placeholder)
    descriptions.append(hist_description)

    with st.beta_expander("Show Descriptive Stats"):
        st.dataframe(df_proteins[["Protein", "NumPeptides"]].describe())
    st.text("")

######################################
# Peptide Data
######################################

if show_pca or show_clustergram:
    df_peptides = get_peptide_data(
        key=f"wide/{dataset}/RESULTS-quant.elib.peptides.txt"
    )
    pca_peptides = PCA(n_components=2)
    pca_components = pca_peptides.fit_transform(df_peptides.values.T)

if show_pca:
    st.header("PCA Plot")
    fig_pca = get_pca_fig(
        pca_components=pca_components,
        labels=[file_to_condition[f] for f in df_peptides.columns],
        color=primary_color,
        explained_variance_ratio=pca_peptides.explained_variance_ratio_,
    )
    st.write(fig_pca)
    figures.append(fig_pca)

    pca_placeholder = "A PCA (Principal Component Analysis) Plot where for each sample/bio replicate the peptide intensity was reduced to two principal components. Samples that are closer together are more similar, samples that are farther apart less so. Most ideally we'll see similar replicates/treatments clustered together. If not, there could have potentially been batch effects. The input data to the PCA algorithm were the raw, unnormalized intensities."
    pca_description = st.text_area("Description", key="pca", value=pca_placeholder)
    descriptions.append(pca_description)

if show_clustergram:
    st.header("Clustergram Plot")
    NUM_PEPTIDES = 150
    st.sidebar.header("Clustergram Plot")
    cluster_selection_method = st.sidebar.selectbox(
        "Sort By", ["PCA Most Influential Peptides", "Chronological"]
    )
    if cluster_selection_method == "Chronological":
        cluster_start = st.sidebar.slider(
            f"Select start of range ({NUM_PEPTIDES} peptides at a time)",
            min_value=0,
            max_value=len(df_peptides) - NUM_PEPTIDES,
        )
        fig_clust = get_clustergram_fig(
            df=df_peptides.iloc[cluster_start : cluster_start + NUM_PEPTIDES, :],
            column_labels=[
                file_to_condition[f] for f in list(df_peptides.columns.values)
            ],
        )
    else:
        pca_dim = st.sidebar.selectbox("PCA Dimension", [i + 1 for i in list(range(2))])
        most_important_features = [
            (-np.abs(comp)).argsort()[:NUM_PEPTIDES]
            for comp in pca_peptides.components_
        ]
        fig_clust = get_clustergram_fig(
            df=df_peptides.iloc[most_important_features[pca_dim - 1], :],
            column_labels=[
                file_to_condition[f] for f in list(df_peptides.columns.values)
            ],
        )
    st.write(fig_clust)
    figures.append(fig_clust)

    cluster_placeholder = f"A clustergram showing the top {NUM_PEPTIDES} log10 peptide intensities (y-axis) for each sample/bio replicate (x-axis) sorted by: {cluster_selection_method}. The dendrograms on each side cluster the data by peptides and sample similarity on the y- and x-axis respectively. The cluster method used is single linkage."
    cluster_description = st.text_area(
        "Description", key="cluster", value=cluster_placeholder
    )
    descriptions.append(cluster_description)


######################################
# Export
######################################

if st.button("Export to PDF"):
    with st.spinner(text="Loading"):
        streamlit_report_to_pdf(
            title=TITLE_TEXT,
            dataset_name=dataset,
            figures=figures,
            descriptions=descriptions,
            write_html=True,
        )
