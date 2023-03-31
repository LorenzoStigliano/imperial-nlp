import os
from typing import Any

from .nlp_model import NLPModel
from .config import PLOT_FOLDER, Array_like

# import plotly
# import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


class Analysis(NLPModel):
    def __init__(self) -> None:
        super().__init__()
        self.plot_folder = PLOT_FOLDER
        self.create_folder(self.plot_folder)

    def __repr__(self) -> str:
        return "%s(%r)" % (self.__class__, self.__dict__)

    def __str__(self) -> str:
        return "%s(%r)" % (self.__class__, self.__dict__)

    def save_figure(self, fig, fig_name: str) -> None:
        fig_path = os.path.join(self.plot_folder, fig_name)
        if isinstance(fig, plotly.graph_objs._figure.Figure):  # plotly
            fig.write_image(fig_path)
        else:  # matplotlib
            fig.savefig(fig_path)

    # Dataset analysis
    def dataset_analysis(self, df_analysis: pd.DataFrame) -> None:
        # make a copy first
        df = df_analysis.copy(deep=True)

        # Label per type of article
        article_type = np.unique(df["keyword"])
        fig = go.Figure(
            data=[
                go.Bar(
                    name="No PCL",
                    x=article_type,
                    y=[
                        len(df[(df["keyword"] == c) & (df["binary_label"] == 0)])
                        for c in article_type
                    ],
                ),
                go.Bar(
                    name="PCL",
                    x=article_type,
                    y=[
                        len(df[(df["keyword"] == c) & (df["binary_label"] == 1)])
                        for c in article_type
                    ],
                ),
            ]
        )
        fig.update_layout(
            barmode="group",
            showlegend=True,
            title_text="Label per article type",
            xaxis_tickangle=-45,
        )
        self.save_figure(fig, "label_article.png")

        # Label per type of article for PCL data
        article_type = np.unique(df["keyword"])
        fig = go.Figure(
            data=go.Bar(
                name="PCL",
                x=article_type,
                y=[
                    len(df[(df["keyword"] == c) & (df["binary_label"] == 1)])
                    for c in article_type
                ],
            )
        )
        fig.update_layout(
            barmode="group",
            showlegend=True,
            title_text="Label per article type for PCL data",
            xaxis_tickangle=-45,
        )
        self.save_figure(fig, "label_article_pcl.png")

        # Label per country code
        country_codes = np.unique(df["country_code"])
        fig = go.Figure(
            data=[
                go.Bar(
                    name="No PCL",
                    x=country_codes,
                    y=[
                        len(df[(df["country_code"] == c) & (df["binary_label"] == 0)])
                        for c in country_codes
                    ],
                ),
                go.Bar(
                    name="PCL",
                    x=country_codes,
                    y=[
                        len(df[(df["country_code"] == c) & (df["binary_label"] == 1)])
                        for c in country_codes
                    ],
                ),
            ]
        )
        fig.update_layout(
            barmode="group",
            showlegend=True,
            title_text="Label per country code",
            xaxis_tickangle=-45,
        )
        self.save_figure(fig, "label_country.png")

        # Length of text and label - Plot and Box and Whisker Plot
        df["length_text"] = df["text"].apply(lambda x: len(x))

        df_test = df.copy(deep=True)
        df_test["length_text"] = df["text"].apply(lambda x: len(x))
        fig, ax = plt.subplots()
        ax.set_title("Length text for no PCL and PCL")
        ax.boxplot(
            [
                df_test[
                    (df_test["binary_label"] == 0) & (df_test["length_text"] < 750)
                ]["length_text"],
                df_test[
                    (df_test["binary_label"] == 1) & (df_test["length_text"] < 750)
                ]["length_text"],
            ]
        )
        ax.set_xticklabels(["no PCL", "PCL"])
        fig.show()
        quantiles = [np.quantile(df["length_text"], i / 10) for i in range(11)]
        fig = go.Figure(
            data=[
                go.Bar(
                    name="No PCL",
                    x=[f"Quantile {int(10 * i)}%" for i in range(1, 11)],
                    y=[
                        len(
                            df[
                                (df["length_text"] >= quantiles[i])
                                & (df["length_text"] <= quantiles[i + 1])
                                & (df["binary_label"] == 0)
                            ]
                        )
                        for i in range(10)
                    ],
                ),
                go.Bar(
                    name="PCL",
                    x=[f"Quantile {int(10 * i)}%" for i in range(1, 11)],
                    y=[
                        len(
                            df[
                                (df["length_text"] >= quantiles[i])
                                & (df["length_text"] <= quantiles[i + 1])
                                & (df["binary_label"] == 1)
                            ]
                        )
                        for i in range(10)
                    ],
                ),
            ]
        )
        fig.update_layout(
            barmode="group",
            showlegend=True,
            title_text="Label per length of text quantile",
            xaxis_tickangle=-45,
        )
        self.save_figure(fig, "label_length_text.png")

    # Model analysis

    # Global
    def run_analysis(self) -> None:
        df_pcl = self.data_pcl
        #self.dataset_analysis(df_pcl)  # raw dataset analysis
        df_pcl_processed = self.run_preprocessing_pcl()
        model = self.run_model()
        print("Analysis done!")
