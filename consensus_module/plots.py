import pandas as pd

import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'iframe'
import os
import img2pdf

class Plot:
    def __init__(self, top_k_rankings, pdf_name, plot_path = "plots"):
        self.top_k_rankings = top_k_rankings
        self.pdf_name = pdf_name
        
        absolute_path = os.path.dirname(__file__)
        self.plots_path = os.path.join(absolute_path, plot_path)
        
        self.export_plot_explanation_rankings()
    
    def export_plot_explanation_rankings(self):
        rankings_plots_img_bytes = []
        for i in self.top_k_rankings:
            rankings = pd.DataFrame(self.top_k_rankings[i])
            plot_img_bytes = self.plot_explanation_rankings_table(i, rankings)
            rankings_plots_img_bytes.append(plot_img_bytes)
        
        rankings_plots_pdf = img2pdf.convert(rankings_plots_img_bytes)
        
        # save to pdf
        if not os.path.exists(self.plots_path):
            os.makedirs(self.plots_path)
        pdf_name = f"{self.plots_path}/{self.pdf_name}"
        with open(pdf_name, "wb") as file:
            file.write(rankings_plots_pdf)
    
    def plot_explanation_rankings_table(self, instance, df):
        positions = ['<b>1</b>', '<b>2</b>', '<b>3</b>','<b>4</b>', '<b>5</b>']
        rank_colors = ['#FFB0B0', '#FFC89F', '#FFF3A0', '#BEFF91', '#A8DDFF']
        instance_info = f"<b>Instance {instance} - {df.forest_pred[0]}</b>"
        
        fig = go.Figure(data=[go.Table(
        columnwidth = [200, 150,300],
        header=dict(
            values=[instance_info, "<b>Rank Position</b>", "SHAP (weight)", "LIME (weight)", "Anchors(weight)", "Our Approach (weight - agreement index)"],
            line_color='darkslategray', fill_color='white',
            align='center', font=dict(color='black', size=14)
        ),
        cells=dict(
            values=[[], positions, df.shap_approach, df.lime_approach, df.anchors_approach, df.our_approach],
            line_color='darkslategray', fill=dict(color=['white', rank_colors]),
            align='center', font=dict(color='black', size=12)
        ))
        ])
        # fig.show(renderer="png", width=1150, height=500)
        fig_formatted = fig.to_image(format="png", width=1150, height=500, engine="kaleido")
        return fig_formatted