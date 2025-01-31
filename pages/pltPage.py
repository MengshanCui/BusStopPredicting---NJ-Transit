from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import r2_score
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sklearn.model_selection import train_test_split
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from matplotlib import transforms
import os

# change color based on residual
def color_mapping(actual, predicted):
    residual = predicted - actual  
    norm = Normalize(vmin=residual.min(), vmax=residual.max() * 1.5)  # range
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    colors = sm.to_rgba(residual)  
    return colors

class pltPage(QWidget):
    def __init__(self):
        super(pltPage, self).__init__()
        
        self.figure, self.axs = plt.subplots(2, 5, figsize=(18, 12))  
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar) 
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.plot()
    
    def plot(self):
        plt.ion()

        # 文件路径
        dataUrl = "hex_dataForTrain.csv"
        modeldataUrl = "modeldata.xlsx"
        # stopUrl = "Bus_stop\\stops.txt"
        
        # directory = 'BusStop/'
        # stopname = 'stops.txt'
        stopUrl = 'effectiveBusStop.csv'
        modelsUrl = "multiple_models.pkl"

        data = pd.read_csv(dataUrl)
        self.len = len(data['lon'])
        # print(data.head())
        # stopData = pd.read_csv(stopUrl)
        # stopData = consolidate_data(directory, stopname, 'stop_id')
        stopData = pd.read_csv(stopUrl)
        # try to read lat and lon columns
        lon_col = [col for col in stopData.columns if 'lon' in col.lower()][0]
        lat_col = [col for col in stopData.columns if 'lat' in col.lower()][0]

        # list data
        sdLon = stopData[lon_col].tolist()
        sdLat = stopData[lat_col].tolist()

        # print(data.head())

        x_train = pd.read_excel(modeldataUrl, sheet_name='x_train')
        x_test = pd.read_excel(modeldataUrl, sheet_name='x_test')
        y_train = pd.read_excel(modeldataUrl, sheet_name='y_train')
        y_test = pd.read_excel(modeldataUrl, sheet_name='y_test')

        with open(modelsUrl, 'rb') as file:
            loaded_models = pickle.load(file)

        # possible models from training
        models = {
            'Ridge': loaded_models['Ridge'],
            'Lasso': loaded_models['Lasso'],
            'Support Vector Regression': loaded_models['Support Vector Regression'],
            'Random Forest': loaded_models['Random Forest']
        }

        axs = self.axs.flatten()

        self.scatter_plots = []

        # each model
        for i, (name, model) in enumerate(models.items()):
            
            y_pred_test = pd.Series(model.predict(x_test))

            min_length = min(len(y_test), len(y_pred_test))
            y_test_truncated = pd.Series(y_test.to_numpy().ravel()[:min_length])
            y_pred_test_truncated = y_pred_test[:min_length]

            score = r2_score(y_test_truncated, y_pred_test_truncated)
            colors_test = color_mapping(y_test_truncated, y_pred_test_truncated)

            # regression scatter plot
            axs[i].scatter(x_test['PopulationSum'], y_test_truncated, c=colors_test, s=50)
            # aligonal line, not useful
            # axs[i].plot([x_test['PopulationSum'].min(), x_test['PopulationSum'].max()], 
            #             [y_test_truncated.min(), y_test_truncated.max()], 'k--', lw=2)
            axs[i].set_title(f"{name} (R² Score: {score:.2f})")
            axs[i].set_xlabel("Population")
            axs[i].set_ylabel("Actual Bus Stops")
            
            # map scatter plot
            y_pred_data = model.predict(data[['PopulationSum', 'stopTimeSum']])
            colors_data = color_mapping(data['busStopCount'], y_pred_data)
            scatter = axs[i + 5].scatter(data['lon'], data['lat'], c=colors_data, s=2)
            self.scatter_plots.append(scatter)  # store in memory for size changing
            axs[i + 5].set_title(f"{name} Prediction")
            axs[i + 5].set_xlabel("lon")
            axs[i + 5].set_ylabel("lat")
            axs[i + 5].set_facecolor('black')

        # hexbin plot
        hb = axs[9].hexbin(data['lon'], data['lat'], C=data['PopulationSum'], gridsize=140, reduce_C_function=np.sum, cmap='viridis')
        axs[9].set_title('Hexbin Plot')
        # show bus stops
        axs[9].scatter(sdLon, sdLat, color='red', s=1, label='Bus Stops')
        axs[9].legend()
        plt.colorbar(hb, ax=axs[9], label='Sum of Population')

        # zoom back to New Jersey
        axs[9].set_xlim(axs[8].get_xlim())
        axs[9].set_ylim(axs[8].get_ylim())

        # hexbin zoom event
        def sync_hexbin_zoom(event):
            if event.inaxes in [axs[5], axs[6], axs[7], axs[8], axs[9]]:  # symc list
                new_xlim = event.inaxes.get_xlim()
                new_ylim = event.inaxes.get_ylim()          # range

                # could be used to sync based on the center point
                x_range = abs(new_xlim[1] - new_xlim[0])
                y_range = abs(new_ylim[1] - new_ylim[0])
                x_center = (new_xlim[0] + new_xlim[1]) / 2
                y_center = (new_ylim[0] + new_ylim[1]) / 2
                yUpper = y_center + x_range/2
                yLower = y_center - x_range/2
                new_ylim = (yLower, yUpper) 


                for ax in (axs[5], axs[6], axs[7], axs[8], axs[9]):  
                    if ax.get_xlim() != new_xlim or ax.get_ylim() != new_ylim:
                        ax.set_xlim(new_xlim)  
                        ax.set_ylim(new_ylim)  
                        
                # getting bigger when zooming in
                scale_factor = 1 / (x_range * y_range)
                for scatter in self.scatter_plots:
                    scatter.set_sizes([2 * scale_factor for _ in range(len(data['lon']))])
                
                self.figure.canvas.draw_idle()  # 更新图形

        self.figure.canvas.mpl_connect('button_release_event', sync_hexbin_zoom)



        

        # all data as 
        axs[4].scatter(data['PopulationSum'], data['busStopCount'], c='blue', s=50)
        # axs[4].plot([data['PopulationSum'].min(), data['PopulationSum'].max()], [data['busStopCount'].min(), data['BusStopsCount'].max()], 'k--', lw=2)
        axs[4].set_title("All Original Data")
        axs[4].set_xlabel("Population")
        axs[4].set_ylabel("Actual Bus Stops")

        # adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])  
        plt.subplots_adjust(top=0.9, hspace=0.3, wspace=0.4) 

        self.canvas.draw()

    # def update_plot(self, bounds):
                
    #     axs = self.figure.axes  # axs
        
    #     for ax in (axs[5], axs[6], axs[7], axs[8], axs[9]):  

    #         # 计算地图中心点
    #         center_x = (bounds['west'] + bounds['east']) / 2
    #         center_y = (bounds['south'] + bounds['north']) / 2

    #         # 放大或缩小操作（模拟）
    #         zoom_factor = 0.9  # 放大0.9倍，缩小1.1倍等

    #         # 计算缩放后的宽度和高度
    #         new_width = (bounds['east'] - bounds['west']) * zoom_factor
    #         new_height = (bounds['north'] - bounds['south']) * zoom_factor

    #         # 创建一个坐标变换
    #         # 注意，`transforms` 可以将坐标进行缩放而不直接修改 `set_xlim` 和 `set_ylim`
    #         ax.set_xlim(center_x - new_width / 2, center_x + new_width / 2)
    #         ax.set_ylim(center_y - new_height / 2, center_y + new_height / 2)

    #         # 创建一个坐标系变换来进行缩放操作
    #         transform = ax.transData

    #         # 通过缩放来改变显示，而不影响初始坐标范围
    #         scale_transform = transforms.Affine2D().scale(zoom_factor, zoom_factor)
    #         ax.transData = scale_transform + transform

    #     x_range = bounds['east'] - bounds['west']
    #     y_range = bounds['north'] - bounds['south']
    #     # getting bigger when zooming in
    #     scale_factor = 1 / (x_range * y_range)
    #     for scatter in self.scatter_plots:
    #         scatter.set_sizes([2 * scale_factor for _ in range(self.len)])
        
    #     self.figure.canvas.draw()