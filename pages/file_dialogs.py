from PyQt5.QtWidgets import QFileDialog

def open_file_dialogCsv(main_window):
    options = QFileDialog.Options()
    file_name, _ = QFileDialog.getOpenFileName(main_window, "Select CSV File", "./population", "CSV Files (*.csv);;All Files (*)", options=options)
    if file_name:
        main_window.browser.page().runJavaScript("getMapBounds();", lambda bounds: main_window.add_population_points(file_name, bounds))

def open_file_dialogShape(main_window):
    options = QFileDialog.Options()
    file_name, _ = QFileDialog.getOpenFileName(main_window, "Select Shape File", "/BusStop", "TXT Files (*.txt);;All Files (*)", options=options)
    if file_name:
        main_window.browser.page().runJavaScript("getMapBounds();", lambda bounds: main_window.add_shape_points(file_name, bounds))

def open_file_dialogXlsx(main_window):
    options = QFileDialog.Options()
    file_name, _ = QFileDialog.getOpenFileName(main_window, "Select XLSX File", "/hospital", "XLSX Files (*.xlsx);;All Files (*)", options=options)
    if file_name:
        main_window.browser.page().runJavaScript("getMapBounds();", lambda bounds: main_window.add_hospital_points(file_name, bounds))

def open_file_dialogGpx(main_window):
    options = QFileDialog.Options()
    file_name, _ = QFileDialog.getOpenFileName(main_window, "Select GPX File", "/shoppingMall", "GPX Files (*.gpx);;All Files (*)", options=options)
    if file_name:
        main_window.browser.page().runJavaScript("getMapBounds();", lambda bounds: main_window.add_mall_points(file_name, bounds))

def open_file_dialogTxt(main_window):
    options = QFileDialog.Options()
    file_name, _ = QFileDialog.getOpenFileName(main_window, "Select Txt File", "/BusStop", "TXT Files (*.txt);;All Files (*)", options=options)
    if file_name:
        main_window.browser.page().runJavaScript("getMapBounds();", lambda bounds: main_window.add_stop_points(file_name, bounds))

def open_file_dialogHeatMap(main_window):
    options = QFileDialog.Options()
    file_name, _ = QFileDialog.getOpenFileName(main_window, "Select Csv HeatMap File", "", "CSV Files (*.csv);;All Files (*)", options=options)
    if file_name:
        main_window.browser.page().runJavaScript("getMapBounds();", lambda bounds: main_window.add_heatMap_points(file_name, bounds))

def open_file_dialogTrainedData(main_window):
    options = QFileDialog.Options()
    file_name, _ = QFileDialog.getOpenFileName(main_window, "Select Trained Data File", "", "CSV Files (*.csv);;All Files (*)", options=options)
    if file_name:
        main_window.browser.page().runJavaScript("getMapBounds();", lambda bounds: main_window.add_trained_points(file_name, bounds))

def open_file_dialogBoundedTrainedData(main_window):
    options = QFileDialog.Options()
    file_name, _ = QFileDialog.getOpenFileName(main_window, "Select Trained Data File", "", "CSV Files (*.csv);;All Files (*)", options=options)
    if file_name:
        main_window.browser.page().runJavaScript("getMapBounds();", lambda bounds: main_window.add_bounded_trained_point(file_name, bounds))

def alls_bounds(main_window):
    main_window.browser.page().runJavaScript("getMapBounds();", lambda bounds: main_window.add_all_stops(bounds))

