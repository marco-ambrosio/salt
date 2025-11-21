import cv2
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QAction,
    QApplication,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QStatusBar,
    QTreeWidget,
    QTreeWidgetItem,
    QHeaderView,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QWheelEvent, QMouseEvent, QCloseEvent
from PyQt5.QtCore import Qt, QRectF


selected_annotations = []


class CustomGraphicsView(QGraphicsView):
    def __init__(self, editor, tracking_mode=False):
        super(CustomGraphicsView, self).__init__()

        self.editor = editor
        self.tracking_mode = tracking_mode

        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setRenderHint(QPainter.TextAntialiasing)

        self.setOptimizationFlag(QGraphicsView.DontAdjustForAntialiasing, True)
        self.setOptimizationFlag(QGraphicsView.DontSavePainterState, True)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setInteractive(True)

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.image_item = None

    def set_image(self, q_img):
        pixmap = QPixmap.fromImage(q_img)
        if self.image_item:
            self.image_item.setPixmap(pixmap)
            self.setSceneRect(QRectF(pixmap.rect()))
        else:
            self.image_item = self.scene.addPixmap(pixmap)
            self.setSceneRect(QRectF(pixmap.rect()))
        # Image scaling: always fill the view, up or down, keeping aspect ratio
        view_size = self.viewport().size()
        img_size = pixmap.size()
        if img_size.width() > 0 and img_size.height() > 0:
            scale_x = view_size.width() / img_size.width()
            scale_y = view_size.height() / img_size.height()
            scale = min(scale_x, scale_y)
            self.resetTransform()
            self.scale(scale, scale)
        else:
            self.resetTransform()
        # Optionally, center the image
        self.centerOn(self.sceneRect().center())

    def wheelEvent(self, event: QWheelEvent):
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            adj = (event.angleDelta().y() / 120) * 0.1
            self.scale(1 + adj, 1 + adj)
        else:
            delta_y = event.angleDelta().y()
            delta_x = event.angleDelta().x()
            x = self.horizontalScrollBar().value()
            self.horizontalScrollBar().setValue(x - delta_x)
            y = self.verticalScrollBar().value()
            self.verticalScrollBar().setValue(y - delta_y)

    def keyPressEvent(self, event):
        adj = 0.1

        # Zoom in/out with Ctrl++ and Ctrl+- (applies only if this is the image view widget)
        if event.modifiers() == Qt.ControlModifier:
            if event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
                self.scale(1 + adj, 1 + adj)
            if event.key() == Qt.Key_Minus:
                self.scale(1 - adj, 1 - adj)
        super().keyPressEvent(event)

    def imshow(self, img):
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.set_image(q_img)

    def mousePressEvent(self, event: QMouseEvent) -> None:

        # FUTURE USE OF RIGHT CLICK EVENT IN THIS AREA
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            print("Control/ Command key pressed during a mouse click")
            pos = event.pos()
            pos_in_item = self.mapToScene(pos) - self.image_item.pos()
            x, y = int(pos_in_item.x()), int(pos_in_item.y())
            if event.button() == Qt.LeftButton:
                print(f"CTRL + Left click at ({x}, {y})")
        else:
            if self.tracking_mode:
                return
            pos = event.pos()
            pos_in_item = self.mapToScene(pos) - self.image_item.pos()
            x, y = pos_in_item.x(), pos_in_item.y()
            if event.button() == Qt.LeftButton:
                label = 1
            elif event.button() == Qt.RightButton:
                label = 0
            self.editor.add_click([int(x), int(y)], label)
            self.imshow(self.editor.display)


class ApplicationInterface(QMainWindow):
    def __init__(self, app, editor, tracking_mode=False, panel_size=(1920, 1080)):
        super().__init__()
        self.app = app
        self.editor = editor
        self.panel_size = panel_size
        self.tracking_mode = tracking_mode

        self.setStatusBar(QStatusBar(self))

        file_menu = self.menuBar().addMenu("&File")
        go_to_image_action = QAction("&Go To Image", self)
        go_to_image_action.setStatusTip("Go to a specific image by ID")
        go_to_image_action.triggered.connect(self.go_to_image)
        file_menu.addAction(go_to_image_action)
        save_action = QAction("&Save", self)
        save_action.setStatusTip("Save the current annotation (Ctrl+S)")
        save_action.triggered.connect(self.save_all)
        file_menu.addAction(save_action)
        exit_action = QAction("&Exit", self)
        exit_action.setStatusTip("Exit the application")
        exit_action.triggered.connect(self.quit)
        file_menu.addAction(exit_action)

        edit_menu = self.menuBar().addMenu("&Edit")
        sort_action = QAction("&Sort Images", self)
        sort_action.setStatusTip("Sort images in the dataset explorer by filenames")
        sort_action.triggered.connect(self.sort_images)
        edit_menu.addAction(sort_action)
        # add_new_category_action = QAction("Add New Category", self)
        # add_new_category_action.setStatusTip("Add a new category to the dataset")
        # add_new_category_action.triggered.connect(self.editor.add_new_category)
        # edit_menu.addAction(add_new_category_action)

        view_menu = self.menuBar().addMenu("&View")
        toggle_action = QAction("&Toggle Annotations", self)
        toggle_action.setStatusTip("Toggle annotation visibility (T)")
        toggle_action.triggered.connect(self.toggle)
        view_menu.addAction(toggle_action)
        transparency_up_action = QAction("Transparency Up", self)
        transparency_up_action.setStatusTip("Increase transparency (L)")
        transparency_up_action.triggered.connect(self.transparency_up)
        view_menu.addAction(transparency_up_action)
        transparency_down_action = QAction("Transparency Down", self)
        transparency_down_action.setStatusTip("Decrease transparency (K)")
        transparency_down_action.triggered.connect(self.transparency_down)
        view_menu.addAction(transparency_down_action)
        # add a toggle for the color by tracker mode
        if self.tracking_mode:
            color_by_tracker_action = QAction("Color by Tracker ID", self)
            color_by_tracker_action.setStatusTip("Toggle color by tracker ID")
            color_by_tracker_action.setCheckable(True)
            color_by_tracker_action.setChecked(False)
            color_by_tracker_action.triggered.connect(lambda: self.toggle_color_by_tracker())
            view_menu.addAction(color_by_tracker_action)

        self.layout = QVBoxLayout()

        self.top_bar = self.get_top_bar()
        self.layout.addWidget(self.top_bar)

        self.main_window = QHBoxLayout()
        if not tracking_mode:
            layout = QVBoxLayout()
            self.main_window.addLayout(layout)
            label = QLabel("Current Image")
            label.setStyleSheet("font-weight: bold; font-size: 16px;")
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label)
            self.graphics_view = CustomGraphicsView(editor)
            layout.addWidget(self.graphics_view)
        else:
            self.images_column = QVBoxLayout()
            label = QLabel("Current Image")
            label.setStyleSheet("font-weight: bold; font-size: 16px;")
            label.setAlignment(Qt.AlignCenter)
            self.images_column.addWidget(label)
            self.graphics_view = CustomGraphicsView(editor)
            self.images_column.addWidget(self.graphics_view)
            label = QLabel("Previous Image (only preview)")
            label.setStyleSheet("font-weight: bold; font-size: 16px;")
            label.setAlignment(Qt.AlignCenter)
            self.images_column.addWidget(label)
            self.next_graphics_view = CustomGraphicsView(editor, tracking_mode=True)
            self.images_column.addWidget(self.next_graphics_view)
            self.main_window.addLayout(self.images_column)

        # Side panel for categories
        self.panel = self.get_side_panel()
        self.main_window.addWidget(self.panel)

        self.panel_annotations = QTreeWidget()
        if tracking_mode:
            self.panel_annotations.setColumnCount(3)
            self.panel_annotations.setHeaderLabels(["ID", "Tracker ID", "Category"])
            self.panel_annotations.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
            self.panel_annotations.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
            self.panel_annotations.header().setSectionResizeMode(2, QHeaderView.Stretch)
        else:
            self.panel_annotations.setColumnCount(2)
            self.panel_annotations.setHeaderLabels(["ID", "Category"])
            self.panel_annotations.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
            self.panel_annotations.header().setSectionResizeMode(1, QHeaderView.Stretch)
        self.panel_annotations.setSelectionMode(QAbstractItemView.MultiSelection)
        self.panel_annotations.setAlternatingRowColors(True)
        self.panel_annotations.setRootIsDecorated(False)
        self.panel_annotations.setIndentation(0)
        self.panel_annotations.setSortingEnabled(True)
        self.panel_annotations.sortByColumn(0, Qt.AscendingOrder)
        self.panel_annotations.itemSelectionChanged.connect(self.annotation_list_item_clicked)
        self.main_window.addWidget(self.panel_annotations)

        self.layout.addLayout(self.main_window)

        widget = QWidget()
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)

        self.update_view()

    def update_view(self):
        self.graphics_view.imshow(self.editor.display)
        self.get_side_panel_annotations()

        if self.tracking_mode:
            self.next_graphics_view.imshow(self.editor.draw_prev_image_with_annotations())

    def closeEvent(self, event):
        self.quit(event)

    def toggle_color_by_tracker(self):
        self.editor.toggle_color_by_tracker()
        self.update_view()

    def confirmation_dialog(self, title, message):
        reply = QMessageBox.question(
            self,
            title,
            message,
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        return reply

    def quit(self, event=None):
        reply = self.confirmation_dialog("Quit", "Do you want to save your changes before quitting?")

        if reply == QMessageBox.Yes:
            self.save_all()
            if isinstance(event, QCloseEvent):
                event.accept()
        elif reply == QMessageBox.Cancel:
            if isinstance(event, QCloseEvent):
                event.ignore()
            return

        if isinstance(event, QCloseEvent):
            event.accept()
        self.app.quit()

    def sort_images(self):
        reply = self.confirmation_dialog(
            "Sort Images",
            "Do you want to sort the images in the dataset explorer by their filenames?",
        )
        if reply == QMessageBox.Yes:
            self.editor.sort_images()
            self.statusBar().showMessage("Images sorted successfully!", 5000)
            self.reset()

    def reset(self):
        self.editor.reset()
        self.panel_annotations.clearSelection()
        self.update_view()

    def add(self):
        self.editor.save_ann()
        self.reset()
        self.statusBar().showMessage("Annotation added successfully!", 5000)

    def next_image(self):
        self.editor.next_image()
        self.reset()

    def prev_image(self):
        self.editor.prev_image()
        self.reset()
    
    def go_to_image(self):
        items = self.editor.dataset_explorer.getImgIds()
        items = [str(i) for i in sorted(items)]
        item, ok = QInputDialog.getItem(self, "Go To Image", "Select Image ID:", items, 0, False)
        if not ok or not item:
            return
        image_id = int(item)
        self.editor.go_to_image(image_id)
        self.reset()

    def toggle(self):
        self.editor.toggle(self.get_selected_annotations())
        self.update_view()

    def transparency_up(self):
        self.editor.step_up_transparency(self.get_selected_annotations())
        self.update_view()

    def transparency_down(self):
        self.editor.step_down_transparency(self.get_selected_annotations())
        self.update_view()

    def save_all(self):
        self.editor.save()
        self.statusBar().showMessage("Annotations saved successfully!", 5000)

    def change_annotation_category(self):
        self.editor.change_annotation_category(self.get_selected_annotations())
        self.reset()
        self.update_view()

    def change_annotation_tracker_id(self):
        selected_annotations, tracker_ids = self.get_selected_annotations(get_tracker_id=True)

        if len(selected_annotations) != 1:
            error_box = QMessageBox()
            error_box.setIcon(QMessageBox.Warning)
            error_box.setText("Please select exactly one annotation to change its ID.")
            error_box.setWindowTitle("Error")
            error_box.setStandardButtons(QMessageBox.Ok)
            error_box.setDefaultButton(QMessageBox.Ok)
            error_box.exec_()
            return

        dialog = QInputDialog(self)
        dialog.setWindowTitle("Change Annotation ID")
        dialog.setLabelText("Enter new ID:")
        dialog.setIntValue(tracker_ids[0] if tracker_ids[0] >= 0 else self.editor.next_tracker_id)
        dialog.setInputMode(QInputDialog.IntInput)
        dialog.setTextEchoMode(QLineEdit.Normal)
        dialog.setOkButtonText("Change")
        dialog.setCancelButtonText("Cancel")
        dialog.exec_()

        try:
            if dialog.result() == QInputDialog.Accepted:
                new_id = dialog.intValue()
                self.editor.change_annotation_tracker_id(selected_annotations, new_id)
                self.statusBar().showMessage(f"Annotation ID changed to {new_id} successfully!", 5000)
            else:
                self.statusBar().showMessage("Annotation ID change cancelled.", 5000)
        except ValueError as e:
            error_box = QMessageBox()
            error_box.setIcon(QMessageBox.Warning)
            error_box.setText(str(e))
            error_box.setWindowTitle("Error")
            error_box.setStandardButtons(QMessageBox.Ok)
            error_box.setDefaultButton(QMessageBox.Ok)
            error_box.exec_()
        finally:
            self.reset()
            self.update_view()

    def get_top_bar(self):
        top_bar = QWidget()
        button_layout = QHBoxLayout(top_bar)
        # suppress warning "QLayout::addChildLayout: layout "" already has a parent"
        # self.layout.addLayout(button_layout)
        buttons = [
            ("Add", lambda: self.add(), "Add the currently selected annotation (N)"),
            ("Reset", lambda: self.reset(), "Reset the current annotation and view (R)"),
            ("Prev", lambda: self.prev_image(), "Go to the previous image (A)"),
            ("Next", lambda: self.next_image(), "Go to the next image (D)"),
            ("Remove Selected Annotations", lambda: self.delete_annotations(), "Delete selected annotations"),
            (
                "Change Category",
                lambda: self.change_annotation_category(),
                "Change the category of the selected annotations",
            ),
            (
                "Change ID",
                lambda: self.change_annotation_tracker_id(),
                "Change the Tracking ID of the selected annotation",
            ),
        ]
        for button, lmb, tooltip in buttons:
            bt = QPushButton(button)
            bt.clicked.connect(lmb)
            bt.setToolTip(tooltip)
            button_layout.addWidget(bt)

        return top_bar

    def get_side_panel(self):
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        categories, colors = self.editor.get_categories(get_colors=True)
        label_array = []
        for i, _ in enumerate(categories):
            label_array.append(QRadioButton(categories[i]))
            label_array[i].clicked.connect(lambda state, x=categories[i]: self.editor.select_category(x))
            label_array[i].setStyleSheet("background-color: rgba({},{},{},0.6)".format(*colors[i][::-1]))
            panel_layout.addWidget(label_array[i])

        scroll = QScrollArea()
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setWidget(panel)
        scroll.setFixedWidth(200)
        return scroll

    def get_side_panel_annotations(self):
        anns, colors = self.editor.list_annotations()
        list_widget = self.panel_annotations
        list_widget.clear()
        categories = self.editor.get_categories(get_colors=False)

        if self.tracking_mode:
            for ann in anns:
                item = QTreeWidgetItem(list_widget)
                item.setText(0, str(ann["id"]))
                tr_id = ann.get("tracker_id", -1)
                tr_id_text = str(tr_id) if tr_id >= 0 else "N/A"
                item.setText(1, tr_id_text)
                item.setText(2, categories[ann["category_id"]])
                item.setData(0, Qt.UserRole, ann["id"])
                item.setData(1, Qt.UserRole, tr_id)
        else:
            for ann in anns:
                item = QTreeWidgetItem(list_widget)
                item.setText(0, str(ann["id"]))
                item.setText(1, categories[ann["category_id"]])
                item.setData(0, Qt.UserRole, ann["id"])

        return list_widget

    def delete_annotations(self):
        for annotation in self.get_selected_annotations():
            self.editor.delete_annotations(annotation)
        self.get_side_panel_annotations()
        self.reset()

    def annotation_list_item_clicked(self):
        selected_annotations = self.get_selected_annotations()
        self.editor.draw_selected_annotations(selected_annotations)
        self.graphics_view.imshow(self.editor.display)

    def get_selected_annotations(self, get_tracker_id=False):
        selected_annotations = []
        if not get_tracker_id:
            for item in self.panel_annotations.selectedItems():
                selected_annotations.append(item.data(0, Qt.UserRole))
            return selected_annotations
        else:
            tracker_ids = []
            for item in self.panel_annotations.selectedItems():
                selected_annotations.append(item.data(0, Qt.UserRole))
                tracker_ids.append(item.data(1, Qt.UserRole))
            return selected_annotations, tracker_ids

    def keyPressEvent(self, event):
        # if event.key() == Qt.Key_Escape:
        #     self.app.quit()
        if event.key() == Qt.Key_A:
            self.prev_image()
        if event.key() == Qt.Key_D:
            self.next_image()
        if event.key() == Qt.Key_K:
            self.transparency_down()
        if event.key() == Qt.Key_L:
            self.transparency_up()
        if event.key() == Qt.Key_N:
            self.add()
        if event.key() == Qt.Key_R:
            self.reset()
        if event.key() == Qt.Key_T:
            self.toggle()
        if event.key() == Qt.Key_C:
            self.change_annotation_category()
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_S:
            self.save_all()
        elif event.key() == Qt.Key_Space:
            print("Space pressed")
            # self.clear_annotations(selected_annotations)
            # Do something if the space bar is pressed
            # pass
