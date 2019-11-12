"""
@author: Srihari
@date: 12/10/2018
@desc: Contains utility functions for visualisation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.ticker import PercentFormatter


def plot_naive_variance(pca, ax):
    '''
    Plots the variance explained by each of the principal components.
    Attributes are not scaled, hence a naive approach.
    
    Parameters
    ----------
    pca: An sklearn.decomposition.pca.PCA instance.
    
    Returns
    -------
    A matplotlib.Axes instance.
    '''
    
    # YOUR CODE HERE
 
    exp_var = pca.explained_variance_ratio_
    sns.lineplot(x=range(len(exp_var)),  y=exp_var, ax=ax)
    ax.set_title("Fraction of Explained Variance")
    ax.set_xlabel("Dimension #")
    ax.set_ylabel("Explained Variance Ratio")
    
    
def plot_pca_var_cum(pca, ax, cutoff=0.95):
    '''
    Plots the cumulative variance explained by each of the principal components.
    
    Parameters
    ----------
    pca: An sklearn.decomposition.pca.PCA instance.
    
    Returns
    -------
    A matplotlib.Axes instance.
    '''
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    sns.lineplot(x=range(len(cum_var)),  y=cum_var, ax=ax)
    plt.axhline(cutoff, 1, 0, c="orange", linestyle="--")
    tmp = list(abs(cum_var-cutoff))
    cx = tmp.index(min(tmp))+1
    plt.axvline(cx, cutoff, 0, c="orange", linestyle="--")
    ax.set_title("Cumulative Explained Variance")
    ax.set_xlabel("Dimension #")
    ax.set_ylabel("Cumulative Explained Variance Ratio")
    return cx


def plot_pareto(df, colname, ax):
    df = df.sort_values(by=colname,ascending=False)
    df["cumpercentage"] = df[colname].cumsum()/df[colname].sum()*100
    ax.bar(df.index, df[colname], color="C0")
    ax2 = ax.twinx()
    ax2.plot(df.index, df["cumpercentage"], color="C1", marker="D", ms=7)
    ax2.yaxis.set_major_formatter(PercentFormatter())
    ax.tick_params(axis="y", colors="C0")
    ax2.tick_params(axis="y", colors="C1")
    
    

# Define some useful functions here
def print_df_cols(df):
    print("Columns : ")
    for c in df.columns:
        print("\t", c, "  -->  ", df[c].dtype)
    print()


def plot_corr_heatmap(corrmat, ax, annotate=False, annot_size=15):
    # plt.imshow(xcorr, cmap='hot')
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corrmat, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    cutsomcmap = sns.diverging_palette(250, 0, as_cmap=True)
    corrheatmap = \
        sns.heatmap(ax=ax, data=corrmat, mask=mask, annot=annotate,
                    linewidths=0.5, cmap=cutsomcmap, annot_kws={"size": annot_size},
                    vmin=-1, vmax=1)
    plt.show()


def plot_pie(data, col_name, ax):
    col_cnt = data[col_name].value_counts()
    g = col_cnt.plot.pie(startangle=90, autopct='%.2f', ax=ax)


def plot_bar_timegraph(x, y, data, ax, highlight_max_min=False,
                       point_plot=True, annot=True,
                       title="", xlabel="", ylabel=""):
    if highlight_max_min:
        clrs = []
        for v in data[y].values:
            if v < data[y].max():
                if v > data[y].min():
                    clrs.append('lightblue')
                else:
                    clrs.append('darksalmon')
            else:
                clrs.append('lightgreen')
        g1 = sns.barplot(x=x, y=y, data=data, ax=ax, palette=clrs)
    else:
        g1 = sns.barplot(x=x, y=y, data=data, ax=ax, color="lightblue")
    if point_plot:
        g1 = sns.pointplot(x=x, y=y, data=data, ax=ax, color="darkblue")
    if annot:
        # Add labels to the plot
        style = dict(size=12, color='darkblue')
        s1 = np.round(data[y].pct_change().values, 2)
        s1[0] = 0
        for idx, row in data.iterrows():
            rx, ry = row[x], row[y]
            ax.text(idx*0.99, ry, str(s1[idx]), **style, va="bottom", ha='right')
    g1.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.set_ylim([0, data[y].max() * 1.2])
    ax.legend(handles=ax.lines[::len(data) + 1], labels=[y, y + " % change"])


def plot_box_timegraph(x, y, data, agg_rule, ax, point_plot=True, annot=False,
                       title="", xlabel="", ylabel=""):
    # Get the median value at each year
    agg_data = data[[y, x]].groupby(by=[x], as_index=False).agg(agg_rule)
    g = sns.boxplot(x=x, y=y, data=data[[y, x]], ax=ax)
    if point_plot:
        g = sns.pointplot(x=x, y=y, data=agg_data, ax=ax, color="k")
    if annot:
        # Add labels to the plot
        style = dict(size=12, color='darkblue')
        s1 = np.round(agg_data[y].values, 2)
        for idx, row in agg_data.iterrows():
            rx, ry = row[x], row[y]
            ax.text(idx, ry, str(s1[idx]), **style, va="bottom", ha='center')

    g.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.set_ylim([0, data[y].max() * 1.2])
    ax.legend(handles=ax.lines[::len(data)], labels=[y + " " + agg_rule])

    
def plot_violin(x, y, data, agg_rule, ax, point_plot=True, annot=False,
                       title="", xlabel="", ylabel="", **kwargs):
    # Get the median value at each year
    agg_data = data[[y, x]].groupby(by=[x], as_index=False).agg(agg_rule)
    g = sns.violinplot(x=x, y=y, data=data, ax=ax, **kwargs)
    if point_plot:
        g = sns.pointplot(x=x, y=y, data=agg_data, ax=ax, color="k")
    if annot:
        # Add labels to the plot
        style = dict(size=12, color='darkblue')
        s1 = np.round(agg_data[y].values, 2)
        for idx, row in agg_data.iterrows():
            rx, ry = row[x], row[y]
            ax.text(idx, ry, str(s1[idx]), **style, va="bottom", ha='center')

    g.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.set_ylim([0, data[y].max() * 1.2])
    ax.legend(handles=ax.lines[::len(data)], labels=[y + " " + agg_rule])
    return g


def plot_bubblehist(x, y, s, data, show_max_min=True, title="", xlabel="", ylabel="", ax=None):
    if ax is None:
        fig_size = (16, 9)
        f, ax = plt.subplots(1, 1, figsize=fig_size)
    else:
        fig_size = ax.figure.get_size_inches()
    bubble_scale = 1 - min(fig_size) / max(fig_size)
    g = sns.scatterplot(x=data[x].values,
                        y=data[y].values,
                        s=data[s] * bubble_scale,
                        alpha=0.4, ax=ax)
    if show_max_min:
        max_x_coords, max_y_coords, max_s_val = [], [], []
        min_x_coords, min_y_coords, min_s_val = [], [], []
        for x1 in data[x].unique():
            for y1 in data[y].unique():
                val = data[(data[x] == x1) & (data[y] == y1)][s].values
                if val < data[data[x] == x1][s].max():
                    if val > data[data[x] == x1][s].min():
                        continue
                    else:
                        sval = data[(data["year"] == x1) & (data[y] == y1)][s].values
                        min_x_coords.append(x1)
                        min_y_coords.append(y1)
                        min_s_val.append(sval * bubble_scale)
                else:
                    sval = data[(data["year"] == x1) & (data[y] == y1)][s].values
                    max_x_coords.append(x1)
                    max_y_coords.append(y1)
                    max_s_val.append(sval * bubble_scale)
        plt.scatter(x=max_x_coords, y=max_y_coords, s=max_s_val, c="green", alpha=0.5)
        plt.plot(max_x_coords, max_y_coords, 'g-.')
        plt.scatter(x=min_x_coords, y=min_y_coords, s=min_s_val, c="red", alpha=0.5)
        plt.plot(min_x_coords, min_y_coords, 'r-.')
        # What is the overall maximum?
        max_idx = max_s_val.index(max(max_s_val))
        min_idx = min_s_val.index(min(min_s_val))
        plt.scatter(x=max_x_coords[max_idx], y=max_y_coords[max_idx],
                    s=max_s_val[max_idx],
                    c="green", alpha=1)
        plt.scatter(x=min_x_coords[min_idx], y=min_y_coords[min_idx],
                    s=min_s_val[min_idx],
                    c="red", alpha=1)
    g.set(xlabel=xlabel, ylabel=ylabel, title=title)


    
def plot_dist(data, colname, xlabel="", ylabel="", title="", legend=True):
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
 
    # Add a graph in each part
    gbox = sns.boxplot(data[colname], ax=ax_box)
    ghist = sns.distplot(data[colname], ax=ax_hist)

    # Remove x axis name for the boxplot
    ax_box.set(xlabel='')

    ghist.set(xlabel=xlabel, ylabel=ylabel, title=title)
    if legend:
        ax_hist.legend(handles=ax_hist.lines[::len(data) + 1], labels=[colname])

    

def plot_bar(data, x, y, ax, title="", xlabel="", ylabel="",
             xrot=0, yrot=0, highlight_max_min=True,
             point_plot=False, annot=True, legend=False):    
    if highlight_max_min:
        clrs = []
        for v in data[y].values:
            if v < data[y].max():
                if v > data[y].min():
                    clrs.append('lightblue')
                else:
                    clrs.append('darksalmon')
            else:
                clrs.append('lightgreen')
        g = sns.barplot(x=x, y=y, data=data, ax=ax, palette=clrs)
    else:
        g = sns.barplot(x=x, y=y, data=data, ax=ax)
    if point_plot:
        g1 = sns.pointplot(x=x, y=y, data=data, ax=ax, color="darkblue")
    if xrot != 0:
        g.set_xticklabels(rotation=xrot, labels=g.get_xticklabels())
    if yrot != 0:
        g.set_yticklabels(rotation=yrot, labels=g.get_yticklabels())
    if annot:
        # Add labels to the plot
        style = dict(size=12, color='darkblue')
        s1 = data[y].values
        counter = 0
        for idx, row in data.iterrows():
            rx, ry = row[x], row[y]
            if type('str') == type(idx):
                ax.text(counter, ry, str(np.round(ry, 2)), 
                    **style, va="bottom", ha='right')
            else:
                ax.text(idx*0.99, ry, str(np.round(s1[idx], 2)), 
                        **style, va="bottom", ha='right')
            counter += 1
    g.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.set_ylim([0, data[y].max() * 1.2])
    if legend:
        ax.legend(handles=ax.lines[::len(data) + 1], labels=[y])


def plot_line(data, x, y, ax, title="", xrot=0, yrot=0, sort_x=False, markers="o"):
    g = sns.lineplot(x=x, y=y, data=data, sort=sort_x, markers=markers, ax=ax)
    if xrot != 0:
        g.set_xticklabels(rotation=xrot, labels=data[x])
    if yrot != 0:
        g.set_yticklabels(rotation=yrot, labels=y)
    plt.title(title)
    plt.show()


def group_and_sort(dataframe, dummycol, groupbycol):
    dataframe = dataframe.join(pd.get_dummies(dataframe[dummycol], dummy_na=False))
    data_grp = dataframe.groupby(by=[groupbycol]).sum()
    data_grp["total"] = data_grp.sum(axis=1)
    data_grp.sort_values(by="total", inplace=True, ascending=False)
    return data_grp.drop("total", axis=1)


def find_common_cols(list_of_cols):
    result = set(list_of_cols[0])
    for s in list_of_cols[1:]:
        result.intersection_update(s)
    return result
