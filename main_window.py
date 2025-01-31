
from PyQt5.QtWidgets import QMainWindow, QStackedWidget, QAction
from pages.mapPage import mapPage as MapPage
from pages.pltPage import pltPage as PltPage
from pages.settingPage import settingPage as SettingsPage




class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.stacked_widget = QStackedWidget()

        self.PltPage = PltPage()
        self.MapPage = MapPage(self.PltPage)
        self.SettingsPage = SettingsPage()

        self.stacked_widget.addWidget(self.MapPage)
        self.stacked_widget.addWidget(self.PltPage)
        self.stacked_widget.addWidget(self.SettingsPage)

        self.setCentralWidget(self.stacked_widget)

        self.create_menu()
        

    def create_menu(self):
        menubar = self.menuBar()

        navigate_menu = menubar.addMenu("Navigate")
        settings_menu = menubar.addMenu("Settings")

        # create actions
        MapPageage_action = QAction("MapPage", self)
        MapPageage_action.triggered.connect(lambda: self.stacked_widget.setCurrentWidget(self.MapPage))

        PltPage_action = QAction("PltPage", self)
        PltPage_action.triggered.connect(lambda: self.stacked_widget.setCurrentWidget(self.PltPage))

        SettingsPage_action = QAction("SettingsPage", self)
        SettingsPage_action.triggered.connect(lambda: self.stacked_widget.setCurrentWidget(self.SettingsPage))

        # add actions to menus
        navigate_menu.addAction(MapPageage_action)
        navigate_menu.addAction(PltPage_action)
        settings_menu.addAction(SettingsPage_action)
