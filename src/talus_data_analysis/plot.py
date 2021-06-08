import dash_bio as dashbio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from matplotlib_venn import venn2, venn3
from scipy.special import binom
from scipy.stats import t


def volcano(
    df,
    log_fold_change_col,
    pvalue_col,
    label_column_name,
    log_fold_change_threshold=(2 ** 1, 2 ** 1),
    pvalue_threshold=(0.05, 0.05),
    filter_labels=None,
    y_limit=10,
    min_x_limit=3,
    title=None,
    dim=(None, None),
    color=("#C8102E", "#A6A6A6", "#308AAD"),
    sign_line=False,
):
    """Creates a Volcano Plot using Plotly.
    Args:
        df (pd.DataFrame): input data frame
        log_fold_change_col (str): The column containing the log fold change.
        pvalue_col (str): The column containing the p-value or adjusted p-value.
        label_column_name (str): The name of the column to use for labelling the relevant points in the plot.
        log_fold_change_threshold (tuple): The log fold change threshold for a value to be statistically significant.
        pvalue_threshold (tuple): The P-value threshold for a value to be statistically significant.
        filter_labels (list): A list of labels to filter for. Only these are going to be labelled in the plot (e.g. a list of protein names)
        y_limit (int): Y-axis limit upwards
        min_x_limit (int): X-axis limit
        title (str): The figure title
        dim (tuple): The plot dimensions (width, height).
        color (tuple): A 3-Tuple of colors with Tuple.0 -> Up-regulated, Tuple.1 -> No regulation and Tuple.2 -> Down-regulated.
        sign_line (bool): Whether of not we want to display a dashed line, showing the threshold cutoff values.

    Returns:
        fig: a Plotly figure
    """
    # keeps the legend consistent with the color inputs
    legend_color_map = {
        color[0]: "Up-regulated",
        color[1]: "No regulation",
        color[2]: "Down-regulated",
    }

    df = df.copy(deep=True)

    # drop columns where p-value is NaN
    df = df[df[pvalue_col].notna()]

    # Set the colors. If a point is outside of the set threshold it's either up or down regulated.
    assert len(set(color)) == 3, "Unique color must be size of 3."
    df.loc[
        (df[log_fold_change_col] >= np.log2(log_fold_change_threshold[0]))
        & (df[pvalue_col] < pvalue_threshold[0]),
        "color",
    ] = color[
        0
    ]  # Up-regulated
    df.loc[
        (df[log_fold_change_col] <= -np.log2(log_fold_change_threshold[1]))
        & (df[pvalue_col] < pvalue_threshold[1]),
        "color",
    ] = color[
        2
    ]  # Down-regulated
    df["color"] = df["color"].fillna(color[1])  # No regulation

    # apply X- and Y-axes limits
    df.loc[df[pvalue_col] < 10 ** (-y_limit), pvalue_col] = 10 ** (-y_limit)

    # setting the X-axis limit to the max of the dataframe (minus NaN and Inf values)
    x_limit = max(
        min_x_limit,
        max(
            np.abs(
                df.loc[
                    (df[log_fold_change_col].notna())
                    & (np.abs(df[log_fold_change_col]) != np.inf),
                    log_fold_change_col,
                ]
            )
        ),
    )
    df.loc[
        (df["issue"].notna())
        & (df["issue"] == "oneConditionMissing")
        & (df[log_fold_change_col] == np.inf),
        log_fold_change_col,
    ] = (
        x_limit - 0.2
    )
    df.loc[
        (df["issue"].notna())
        & (df["issue"] == "oneConditionMissing")
        & (df[log_fold_change_col] == -np.inf),
        log_fold_change_col,
    ] = (x_limit - 0.2) * (-1)

    # TODO: do we want to use np.log2 optionally as well?
    log_pvalue_col = f"log_{pvalue_col}"
    df[log_pvalue_col] = -(np.log10(df[pvalue_col]))

    # copies the name into a new column to use for labelling the points
    display_name = f"{label_column_name}_display"
    df[display_name] = df[label_column_name]

    # if filter_labels is given, show only the labels of the given names, else show only the statistically significant ones
    if filter_labels:
        df.loc[~df[label_column_name].isin(set(filter_labels)), display_name] = ""
    else:
        df.loc[
            (df["color"] == color[1])
            | ((df["issue"].notna()) & (df["issue"] == "oneConditionMissing")),
            display_name,
        ] = ""

    fig = px.scatter(
        data_frame=df,
        x=log_fold_change_col,
        y=log_pvalue_col,
        color="color",
        color_discrete_map="identity",
        hover_name=label_column_name,
        hover_data={
            label_column_name: False,
            display_name: False,
            log_fold_change_col: ":.2f",
            log_pvalue_col: ":.2f",
        },
        text=display_name,
        size=log_pvalue_col,
        size_max=10,
        title=title,
        width=dim[0],
        height=dim[1],
    )

    # Create x and y axis titles
    fig.update_layout(
        xaxis_title="log2 fold change", yaxis_title="−log10 (adjusted p−value)"
    )

    # add a legend to the plot using the color map defined at the start of the function
    for i in range(len(fig["data"])):
        fig["data"][i]["name"] = legend_color_map.get(fig["data"][i]["marker"]["color"])
        fig["data"][i]["showlegend"] = True

    # create cutoff lines for log fold change and p-value threshold
    if sign_line:
        fig.add_hline(
            y=-np.log10(pvalue_threshold[0]),
            line_width=1,
            line_dash="dot",
            line_color="#7d7d7d",
        )
        fig.add_vline(
            x=np.log2(log_fold_change_threshold[0]),
            line_width=1,
            line_dash="dot",
            line_color="#7d7d7d",
        )
        fig.add_vline(
            x=-np.log2(log_fold_change_threshold[1]),
            line_width=1,
            line_dash="dot",
            line_color="#7d7d7d",
        )

    return fig


def volcano_list_presence_absence(df, log_fold_change_col):
    """Creates a Volcano Plot using Plotly.
    Args:
        df (pd.DataFrame): input data frame
        log_fold_change_col (str): The column containing the log fold change.

    Returns:
        presence_absence_df: a dataframe containing the presence and absence proteins
    """
    presence = df[
        (df["issue"].notna())
        & (df["issue"] == "oneConditionMissing")
        & (df[log_fold_change_col] == np.inf)
    ]["ProteinName"].to_list()
    absence = df[
        (df["issue"].notna())
        & (df["issue"] == "oneConditionMissing")
        & (df[log_fold_change_col] == -np.inf)
    ]["ProteinName"].to_list()
    df = pd.DataFrame.from_dict(
        {"Presence": presence, "Absence": absence}, orient="index"
    )
    df = df.transpose()
    df = df.fillna("")
    return df


def heatmap(
    df,
    log_fold_change_col,
    pvalue_col,
    x_label_column_name,
    y_label_column_name,
    value_column_name,
    log_fold_change_threshold=(2 ** 1, 2 ** 1),
    pvalue_threshold=(0.05, 0.05),
    sort_ascending=None,
    filter_labels=[],
    start_idx=0,
    length=50,
    y_limit=10,
    title=None,
    dim=(None, None),
    color=("#308AAD", "#001425", "#C8102E"),
):
    """Creates a Heatmap Plot using Plotly.
    Args:
        df (pd.DataFrame): input data frame
        log_fold_change_col (str): The column containing the log fold change.
        pvalue_col (str): The column containing the p-value or adjusted p-value.
        y_label_column_name (str): The name of the column to use for the Y-axis of the plot.
        x_label_column_name (str): The name of the column to use for the X-axis of the plot.
        value_column_name (str): The name of the column to use for the values (Z-axis) of the plot.
        log_fold_change_threshold (tuple): The log fold change threshold for a value to be statistically significant.
        pvalue_threshold (tuple): The P-value threshold for a value to be statistically significant.
        sort_ascending (bool): Whether we want to sort the input list. If None, don't sort if True sort ascending, if False sort descending.
        filter_labels (list): A list of tuples containing label name and values to filter for. Only these are going to be labelled in the plot (e.g. a list of drug/drug interactions)
        start_idx (int): Where to start in the given data frame (only shows 'length' rows)
        length (int): How many rows to show at once
        y_limit (int): Y-axis limit upwards
        title (str): The figure title
        dim (tuple): The plot dimensions (width, height).
        color (tuple): A 3-Tuple of colors with Tuple.0 -> Up-regulated, Tuple.1 -> No regulation and Tuple.2 -> Down-regulated.

    Returns:
        fig: a Plotly figure
    """
    df = df.copy(deep=True)

    # apply X- and Y-axes limits
    df.loc[df[pvalue_col] < 10 ** (-y_limit), pvalue_col] = 10 ** (-y_limit)

    # setting the log fold change limit to the max of the dataframe (minus NaN and Inf values)
    lfc_limit = max(
        np.abs(
            df.loc[
                (df[log_fold_change_col].notna())
                & (np.abs(df[log_fold_change_col]) != np.inf),
                log_fold_change_col,
            ]
        )
    )
    df.loc[
        (df["issue"].notna())
        & (df["issue"] == "oneConditionMissing")
        & (df[log_fold_change_col] == np.inf),
        log_fold_change_col,
    ] = (
        lfc_limit - 0.2
    )
    df.loc[
        (df["issue"].notna())
        & (df["issue"] == "oneConditionMissing")
        & (df[log_fold_change_col] == -np.inf),
        log_fold_change_col,
    ] = (lfc_limit - 0.2) * (-1)

    # filter for only the relevant values
    df = df[
        (np.abs(df[log_fold_change_col]) >= np.log2(log_fold_change_threshold[0]))
        & (df[pvalue_col] < pvalue_threshold[0])
    ]

    if value_column_name == pvalue_col:
        log_pvalue_col = f"log_{pvalue_col}"
        df[log_pvalue_col] = -(np.log10(df[pvalue_col])) * np.sign(
            df[log_fold_change_col]
        )
        value_column_name = log_pvalue_col

    for (name, values) in filter_labels:
        if values:
            df = df[df[name].isin(set(values))]

    df = df[[x_label_column_name, y_label_column_name, value_column_name]].dropna()
    df = df.pivot(
        index=y_label_column_name, columns=x_label_column_name, values=value_column_name
    )

    if sort_ascending is not None:
        df["sum"] = df.abs().sum(axis=1)
        df = df.sort_values(by="sum", ascending=sort_ascending)
        df = df.drop("sum", axis=1)

    fig = px.imshow(
        df.iloc[start_idx : start_idx + 50],
        width=dim[0],
        height=dim[1],
        color_continuous_scale=color,
        title=title,
        # zmin=-1,
        # zmax=1,
        aspect="auto",
    )

    return fig


def comparison(
    df,
    log_fold_change_col,
    pvalue_col,
    label_column_name,
    pvalue_threshold,
    title=None,
    xaxis_title=None,
    yaxis_title=None,
    color="#001425",
    dim=(None, None),
):
    """Creates a Comparison Plot using Plotly.
    Args:
        df (pd.DataFrame): input data frame
        log_fold_change_col (str): The column containing the log fold change.
        pvalue_col (str): The column containing the p-value or adjusted p-value.
        label_column_name (str): The name of the column to use for labelling the relevant points in the plot.
        pvalue_threshold (float): The P-value threshold for a value to be statistically significant.
        title (str): The figure title
        xaxis_title (str): X-axis title
        yaxis_title (str): Y-axis title
        dim (tuple): The plot dimensions (width, height).
        color (tuple): A 3-Tuple of colors with Tuple.0 -> Up-regulated, Tuple.1 -> No regulation and Tuple.2 -> Down-regulated.
        sign_line (bool): Whether of not we want to display a dashed line, showing the threshold cutoff values.

    Returns:
        fig: a Plotly figure
    """
    df = df.copy(deep=True)

    # drop columns where p-value is NaN
    df = df[df[pvalue_col].notna()]

    df["comp"] = t.ppf(1 - pvalue_threshold / 2, df["DF"]) * df["SE"]

    df["color"] = color

    fig = px.scatter(
        df,
        x=label_column_name,
        y=log_fold_change_col,
        error_y="comp",
        error_y_minus="comp",
        color="color",
        color_discrete_map="identity",
        title=title,
        width=dim[0],
        height=dim[0],
    )

    # Create x and y axis titles
    if xaxis_title:
        fig.update_layout(xaxis_title=xaxis_title)
    if yaxis_title:
        fig.update_layout(yaxis_title=yaxis_title)

    # add a legend to the plot using the color map defined at the start of the function
    for i in range(len(fig["data"])):
        fig["data"][i]["showlegend"] = False

    return fig


def correlation(
    x,
    y,
    filter_outliers=False,
    dim=(None, None),
    scatter_color="black",
    title=None,
    xaxis_title=None,
    yaxis_title=None,
    trendline_color="#001425",
):
    """Creates a Correlation Plot using Plotly.
    Args:
        x (list): The X values
        y (list): The Y values
        filter_outliers (bool): Whether we want to filter outliers
        dim (tuple): The plot dimensions (width, height).
        scatter_color (str): Color to use for the scatter plot
        title (str): The figure title
        xaxis_title (str): X-axis title
        yaxis_title (str): Y-axis title
        trendline_color (str): Color to use for the trendline

    Returns:
        fig: a Plotly figure
    """

    epsilon = 1e-5
    df = pd.DataFrame({"x": np.log10(x + epsilon), "y": np.log10(y + epsilon)})

    if filter_outliers:
        df = df[df > 0.0]

    df["scatter_color"] = scatter_color

    fig = px.scatter(
        data_frame=df,
        x="x",
        y="y",
        color="scatter_color",
        color_discrete_map="identity",
        trendline="ols",
        trendline_color_override=trendline_color,
        title=title,
        width=dim[0],
        height=dim[0],
    )

    # Create x and y axis titles
    if xaxis_title:
        fig.update_layout(xaxis_title=xaxis_title)
    if yaxis_title:
        fig.update_layout(yaxis_title=yaxis_title)

    return fig


def scatter_matrix(
    df,
    x_label_column_name,
    y_label_column_name,
    value_column_name,
    filter_outliers=False,
    log_scaling=True,
    title=None,
    color="#001425",
    opacity=None,
    dim=(None, None),
):
    """Creates a Scatter Matrix Plot using Plotly.
    Args:
        df (pd.DataFrame): input data frame
        x_label_column_name (str): The name of the column to use for the X-axis of the plot.
        y_label_column_name (str): The name of the column to use for the Y-axis of the plot.
        value_column_name (str): The name of the column to use for the values (Z-axis) of the plot.
        filter_outliers (bool): Whether we want to filter outliers
        log_scaling (bool): Whether to use log scaling
        title (str): The figure title
        dim (tuple): The plot dimensions (width, height).
        color (str): Color to use for the scatter plot
        color (float): Opacity of the points in the plot (default: None)

    Returns:
        fig: a Plotly figure
    """
    df = df.copy(deep=True)

    df = df[df[value_column_name].notna()]
    dimensions = df[x_label_column_name].unique()
    df = df.pivot_table(
        index=y_label_column_name, columns=x_label_column_name, values=value_column_name
    )

    if log_scaling:
        epsilon = 0
        if not filter_outliers:
            epsilon = 1e-5

        df = np.log10(df + epsilon)

    df["color"] = color

    fig = px.scatter_matrix(
        df,
        dimensions=dimensions,
        color="color",
        color_discrete_map="identity",
        opacity=opacity,
        title=title,
        width=dim[0],
        height=dim[1],
    )

    return fig


def histogram(
    df,
    x_label_column_name,
    y_label_column_name=None,
    value_cutoff=None,
    color="#001425",
    title=None,
    xaxis_title=None,
    yaxis_title=None,
    dim=(None, None),
):
    """Creates a Histogram Plot using Plotly.
    Args:
        df (pd.DataFrame): input data frame
        x_label_column_name (str): The name of the column to use for the X-axis of the plot.
        y_label_column_name (str): The name of the column to use for the Y-axis of the plot.
        value_cutoff (int): Where to cutoff the values of the given axis
        color (str): Color to use for the scatter plot
        title (str): The figure title
        xaxis_title (str): X-axis title
        yaxis_title (str): Y-axis title
        dim (tuple): The plot dimensions (width, height).

    Returns:
        fig: a Plotly figure
    """
    df = df.copy(deep=True)

    if value_cutoff:
        df.loc[
            df[x_label_column_name] > value_cutoff, x_label_column_name
        ] = value_cutoff

    if color not in df.columns:
        df["color"] = color
        color = "color"
        color_discrete_map = "identity"
    else:
        color_discrete_map = None

    fig = px.histogram(
        df,
        x=x_label_column_name,
        y=y_label_column_name,
        color=color,
        color_discrete_map=color_discrete_map,
        title=title,
        width=dim[0],
        height=dim[1],
    )

    fig.update_layout(
        bargap=0.05,
        bargroupgap=0.05,
    )
    if xaxis_title:
        fig.update_layout(xaxis_title=xaxis_title)
    if yaxis_title:
        fig.update_layout(yaxis_title=yaxis_title)

    return fig


def box(
    df,
    x_label_column_name,
    y_label_column_name,
    log_scaling=False,
    filter_outliers=False,
    show_points="outliers",
    title=None,
    xaxis_title=None,
    yaxis_title=None,
    color="#001425",
    dim=(None, None),
):
    """Creates a Box Plot using Plotly.
    Args:
        df (pd.DataFrame): input data frame
        y_label_column_name (str): The name of the column to use for the Y-axis of the plot.
        x_label_column_name (str): The name of the column to use for the X-axis of the plot.
        log_scaling (bool): Whether to use log scaling
        filter_outliers (bool): Whether we want to filter outliers
        show_points (str|bool): Whether we want to show the 'outlier' points or not.
                                For more, check the 'points' parameter at:
                                https://plotly.github.io/plotly.py-docs/generated/plotly.express.box.html
        dim (tuple): The plot dimensions (width, height).
        title (str): The figure title
        xaxis_title (str): X-axis title
        yaxis_title (str): Y-axis title
        color (tuple): A 3-Tuple of colors with Tuple.0 -> Up-regulated, Tuple.1 -> No regulation and Tuple.2 -> Down-regulated.

    Returns:
        fig: a Plotly figure
    """
    df = df.copy(deep=True)

    df = df[df[y_label_column_name].notna()]

    if log_scaling:
        log_value_col = f"log_{y_label_column_name}"
        # only scale the values that are not 0
        df[log_value_col] = df[y_label_column_name]
        df.loc[df[log_value_col] > 0.0, log_value_col] = np.log2(
            df[df[log_value_col] > 0.0][log_value_col]
        )
        y_label_column_name = log_value_col

    if filter_outliers:
        df = df[df[y_label_column_name] >= 0.0]

    df["color"] = color

    fig = px.box(
        df,
        x=x_label_column_name,
        y=y_label_column_name,
        points=show_points,
        color="color",
        color_discrete_map="identity",
        title=title,
        width=dim[0],
        height=dim[0],
    )

    if xaxis_title:
        fig.update_layout(xaxis_title=xaxis_title)
    if yaxis_title:
        fig.update_layout(yaxis_title=yaxis_title)

    # sort x-axis alphabetically
    fig.update_layout(xaxis={"categoryorder": "category ascending"})

    return fig


def pca(
    pca_components,
    labels,
    dim=(None, None),
    color="#001425",
    title=None,
    xaxis_title=None,
    yaxis_title=None,
):
    """Creates a PCA Plot using Plotly.
    Args:
        pca_components (np.array): pca components from scikit learns pca.fit_transform(value)
        labels (list): a list of labels for each point
        dim (tuple): The plot dimensions (width, height).
        color (str): Color to use for the pca plot
        title (str): The figure title
        xaxis_title (str): X-axis title
        yaxis_title (str): Y-axis title

    Returns:
        fig: a Plotly figure
    """
    df_components = pd.DataFrame(
        pca_components, columns=["pc1", "pc2"], index=labels
    ).reset_index()
    df_components["color"] = color

    fig = px.scatter(
        data_frame=df_components,
        x="pc1",
        y="pc2",
        color="color",
        color_discrete_map="identity",
        text="index",
        title=title,
        width=dim[0],
        height=dim[0],
    )

    # Create x and y axis titles
    if xaxis_title:
        fig.update_layout(xaxis_title=xaxis_title)
    if yaxis_title:
        fig.update_layout(yaxis_title=yaxis_title)

    return fig


def clustergram(
    df,
    column_labels,
    log_scaling=True,
    hide_row_labels=True,
    title=None,
    cluster="all",
    xaxis_title=None,
    yaxis_title=None,
    dim=(None, None),
):
    """Creates a Clustergram Plot using Dash Bio and Plotly.
    Args:
        df (pd.DataFrame): the input dataframe
        column_labels (list): a list of labels for each point
        log_scaling (bool): Whether to use log scaling
        hide_row_labels (bool): whether to hide the row labels
        title (str): The figure title
        dim (tuple): The plot dimensions (width, height).

    Returns:
        fig: a Plotly figure
    """
    hidden_labels = []
    if hide_row_labels:
        hidden_labels.append("row")

    if log_scaling:
        epsilon = 1e-5
        df = np.log10(df + epsilon)

    fig = dashbio.Clustergram(
        data=df,
        color_threshold={"row": 150, "col": 700},
        column_labels=column_labels,
        row_labels=list(df.index),
        hidden_labels=hidden_labels,
        cluster=cluster,
        width=dim[0],
        height=dim[1],
        color_map=[
            [0.0, "#C8102E"],
            [0.33, "#96D8D8"],
            [0.66, "#308AAD"],
            [1.0, "#001425"],
        ],
        line_width=2,
    )
    fig.layout.plot_bgcolor = "#fff"
    fig.layout.paper_bgcolor = "#fff"

    if title:
        fig.layout.title = title

    # sort x-axis alphabetically
    for k in fig.layout:
        if "xaxis" in k and fig.layout[k]["showticklabels"]:
            fig.layout[k]["ticktext"] = sorted(fig.layout[k]["ticktext"])

    return fig


def venn(
    sets, labels=None, dim=(None, None), title=None, colors=["#001425", "#308AAD"]
):
    """Creates a Venn Diagram Overlap Plot using Matplotlib and Plotly.
    Args:
        sets (list): sets to plot overlap for
        labels (list): label for each set
        dim (tuple): The plot dimensions (width, height).
        colors ([str]: Color to use for each set in the plot
        title (str): The figure title

    Returns:
        fig: a Plotly figure
    """
    n_sets = len(sets)

    # Choose and create matplotlib venn diagramm
    if n_sets == 2:
        if labels and len(labels) == n_sets:
            venn_diagram = venn2(sets, labels)
        else:
            venn_diagram = venn2(sets)
    elif n_sets == 3:
        if labels and len(labels) == n_sets:
            venn_diagram = venn3(sets, labels)
        else:
            venn_diagram = venn3(sets)
    # Supress output of venn diagramm
    plt.close()

    # Create empty lists to hold shapes and annotations
    shapes = []
    annotations = []

    # Create empty list to make hold of min and max values of set shapes
    x_max = []
    y_max = []
    x_min = []
    y_min = []

    for i in range(0, n_sets):
        # create circle shape for current set
        shape = go.layout.Shape(
            type="circle",
            xref="x",
            yref="y",
            x0=venn_diagram.centers[i][0] - venn_diagram.radii[i],
            y0=venn_diagram.centers[i][1] - venn_diagram.radii[i],
            x1=venn_diagram.centers[i][0] + venn_diagram.radii[i],
            y1=venn_diagram.centers[i][1] + venn_diagram.radii[i],
            fillcolor=colors[i],
            line_color=colors[i],
            opacity=0.75,
        )

        shapes.append(shape)

        # create set label for current set
        anno_set_label = go.layout.Annotation(
            xref="x",
            yref="y",
            x=venn_diagram.set_labels[i].get_position()[0],
            y=venn_diagram.set_labels[i].get_position()[1],
            text=venn_diagram.set_labels[i].get_text(),
            showarrow=False,
        )

        annotations.append(anno_set_label)

        # get min and max values of current set shape
        x_max.append(venn_diagram.centers[i][0] + venn_diagram.radii[i])
        x_min.append(venn_diagram.centers[i][0] - venn_diagram.radii[i])
        y_max.append(venn_diagram.centers[i][1] + venn_diagram.radii[i])
        y_min.append(venn_diagram.centers[i][1] - venn_diagram.radii[i])

    # determine number of subsets
    n_subsets = sum([binom(n_sets, i + 1) for i in range(0, n_sets)])

    for i in range(0, int(n_subsets)):
        if venn_diagram.subset_labels[i]:
            # create subset label (number of common elements for current subset
            anno_subset_label = go.layout.Annotation(
                xref="x",
                yref="y",
                x=venn_diagram.subset_labels[i].get_position()[0],
                y=venn_diagram.subset_labels[i].get_position()[1],
                text=venn_diagram.subset_labels[i].get_text(),
                showarrow=False,
            )
            annotations.append(anno_subset_label)

    # define off_set for the figure range
    off_set = 0.2

    # get min and max for x and y dimension to set the figure range
    x_max_value = max(x_max) + off_set
    x_min_value = min(x_min) - off_set
    y_max_value = max(y_max) + off_set
    y_min_value = min(y_min) - off_set

    # create plotly figure
    fig = go.Figure()

    # set xaxes range and hide ticks and ticklabels
    fig.update_xaxes(range=[x_min_value, x_max_value], showticklabels=False, ticklen=0)

    # set yaxes range and hide ticks and ticklabels
    fig.update_yaxes(
        range=[y_min_value, y_max_value],
        scaleanchor="x",
        scaleratio=1,
        showticklabels=False,
        ticklen=0,
    )

    # set figure properties and add shapes and annotations
    fig.update_layout(
        plot_bgcolor="white",
        margin=dict(b=0, l=10, pad=0, r=10, t=40),
        width=dim[0],
        height=dim[1],
        shapes=shapes,
        annotations=annotations,
        title=dict(text=title, x=0.5, xanchor="center"),
    )

    return fig
