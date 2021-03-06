import math
import os
import sys
from typing import cast

import numpy as np
import pygame as pg
import pygame_gui as pgui
import torch
import torchvision
from model.model import CAModel
from numpy import typing as npt

import demo.demo_utils as dutils


class Demo:
    """
    Interactive demonstration of the trained neural cellular automata.
    """

    def __init__(self, start: bool = True, safe_frames: bool = True) -> None:
        """
        Constructor.

        Args:
            start (bool, optional): If True, the demo starts automatically. Otherwise run() has to
            be called to start the demo. Defaults to True.
        """
        self.safe_frames = safe_frames

        self.n_channels = 16
        self.n_cols = 28
        self.n_rows = 28

        torch.no_grad()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.world = torch.zeros(1, self.n_channels, self.n_cols, self.n_rows).to(self.device)
        # TODO Do we need to disable gradients or caching?

        self.models = dutils.create_models()
        self.model_name = self.models[0].name.lower()
        self.model_type = self.models[0].modeltypes[0].lower()

        self.isOkay = False

        self.init_pygame()
        if start:
            self.run()

    def init_pygame(self) -> None:
        """
        Initializes all pygame modules, sets up the window and GUI.
        """
        pg.init()

        self.setup_window()

        self.fps = 30
        self.clock = pg.time.Clock()
        self.frame_count = 0

        self.angle = 0

        self.setup_gui()

    def setup_window(self):
        """
        Sets up the pygame window and background.
        """
        self.cell_size = 20

        self.canvas_width = self.n_cols * self.cell_size
        self.canvas_height = self.n_rows * self.cell_size
        self.sidebar_width = 400
        self.bottombar_height = 50
        self.width = self.canvas_width + self.sidebar_width
        self.height = self.canvas_height + self.bottombar_height

        self.window = pg.display.set_mode((self.width, self.height))
        pg.display.set_caption("Neural Cellular Automaton")

        # Background
        self.background = pg.Surface((self.width, self.height))
        self.background.fill((20, 20, 20))

    def setup_gui(self):
        """
        Sets up and places the GUI elements.
        """
        half_sidebar_width = self.sidebar_width / 2

        medium_width = self.sidebar_width - 100
        small_width = half_sidebar_width - 60
        medium_height = 50
        # large_height = 100
        text_height = 30

        x_pos_1 = self.canvas_width + 50
        x_pos_2 = self.canvas_width + half_sidebar_width + 10

        self.ui_manager = pgui.UIManager((self.width, self.height))

        self.status_label = pgui.elements.UILabel(
            pg.Rect((0, self.canvas_height), (self.canvas_width, self.bottombar_height)),
            "",
            self.ui_manager,
        )

        self.model_selection_label = pgui.elements.UILabel(
            pg.Rect((x_pos_1, 20), (medium_width, text_height)),
            "Choose the model and type",
            self.ui_manager,
        )

        models = self.models
        model_names = dutils.get_model_names(models)
        self.model_name_selection = pgui.elements.UIDropDownMenu(
            model_names, model_names[0], pg.Rect((x_pos_1, 50), (small_width, 35)), self.ui_manager,
        )

        # TODO: potentially refactor to be not ugly
        self.model_type_pos = [(x_pos_1, 120), (small_width, 35)]
        self.model_type_selection = pgui.elements.UIDropDownMenu(
            models[0].modeltypes,
            models[0].modeltypes[0],
            pg.Rect((x_pos_1, 120), (small_width, 35)),
            self.ui_manager,
        )

        img = pg.image.load(os.path.join("demo", "trained_models", "blood0x5.png"))
        self.model_target_image = pgui.elements.UIImage(
            pg.Rect((x_pos_2, 50), (28 * 5, 28 * 5)), img, self.ui_manager,
        )

        self.control_label = pgui.elements.UILabel(
            pg.Rect((x_pos_1, 210), (medium_width, text_height)),
            "Reset or clear the canvas",
            self.ui_manager,
        )
        self.reset_button = pgui.elements.UIButton(
            pg.Rect((x_pos_1, 240), (small_width, medium_height)), "Reset", self.ui_manager,
        )
        self.clear_button = pgui.elements.UIButton(
            pg.Rect((x_pos_2, 240), (small_width, medium_height)), "Clear", self.ui_manager,
        )

        self.pause_label = pgui.elements.UILabel(
            pg.Rect((x_pos_1, 310), (medium_width, text_height)),
            "Pause and unpause the demo",
            self.ui_manager,
        )
        self.pause_button = pgui.elements.UIButton(
            pg.Rect((x_pos_1, 340), (medium_width, medium_height)), "Pause", self.ui_manager,
        )

        self.fps_label = pgui.elements.UILabel(
            pg.Rect((x_pos_1, 410), (small_width, text_height)),
            f"FPS: {self.fps}",
            self.ui_manager,
        )
        self.fps_slider = pgui.elements.UIHorizontalSlider(
            pg.Rect((x_pos_1, 440), (small_width, medium_height)),
            30,
            (5, 60),
            self.ui_manager,
            click_increment=5,
        )

        self.angle_label = pgui.elements.UILabel(
            pg.Rect((x_pos_2, 410), (small_width, text_height)),
            f"Angle: {self.angle}??",
            self.ui_manager,
        )
        self.angle_slider = pgui.elements.UIHorizontalSlider(
            pg.Rect((x_pos_2, 440), (small_width, medium_height)),
            0,
            (0, 360),
            self.ui_manager,
            click_increment=30,
        )

        # TODO Maybe add other variations of the models (other filters or something)

    def run(self) -> None:
        """
        Starts the demo and processes the pygame loop.
        """
        self.load_model()

        self.running = True
        self.paused = False

        # Make sure, this array is initialized
        colors = np.zeros((self.n_rows, self.n_cols, 4), dtype=int)

        # Main loop
        while self.running:
            time_delta = self.clock.tick(self.fps) / 1000

            self.handle_events()

            # Draw background
            self.window.blit(self.background, (0, 0))

            # Update world
            if not self.paused:
                colors = self.step()

                if self.safe_frames:
                    torchvision.utils.save_image(
                        self.world[0, :4], f"demo/frames/{self.frame_count}.png"
                    )

            # Draw image
            self.draw_pixels(colors)

            # Print status
            self.status_label.set_text(
                f"FPS: {self.clock.get_fps():>5.2f}  |  step: {self.frame_count}"
            )

            # Update GUI
            self.ui_manager.update(time_delta)
            self.ui_manager.draw_ui(self.window)

            # Draw image border
            pg.draw.rect(
                self.window,
                (200, 200, 200),
                (0, 0, self.canvas_width, self.canvas_height),
                width=2,
            )
            pg.draw.rect(self.window, (200, 200, 200), (770, 50, 140, 140), width=1)

            # Draw everthing to screen
            pg.display.flip()

    def load_model(self) -> None:
        """
        Loads a trained model, spawns a seed and updates the target image in the GUI.

        Args:
            model (str): Name of the trained model, e.g. 'blood0'.
        """
        self.reset()

        filter_type = "sobel"
        if self.model_type == "gauss-laplace":
            filter_type = "gauss_and_laplace"
        elif self.model_type == "fixed-random":
            filter_type = "fixed_random"

        self.model = CAModel(
            n_channels=16,
            hidden_channels=128,
            fire_rate=0.5,
            device=self.device,
            filter_type=filter_type,
        )
        self.model.load_state_dict(
            torch.load(
                os.path.join("demo", "trained_models", f"{self.model_name}_{self.model_type}.pt"),
                map_location=self.device,
            )
        )
        self.model.eval()
        img = pg.image.load(os.path.join("demo", "trained_models", f"{self.model_name}x5.png"))
        self.model_target_image.set_image(img)

    def reset(self, spawn_seed: bool = True) -> None:
        """
        Clears the canvas and spawns a seed.

        Args:
            spawn_seed (bool, optional): If False, no seed is spawned. Defaults to True.
        """
        self.world = torch.zeros(1, self.n_channels, self.n_cols, self.n_rows)
        if spawn_seed:
            self.world[0, 3:, 14, 14] = 1

        self.frame_count = 0

    def handle_events(self):
        """
        Handling of all pygame events since last call. This includes closing the window, mouse
        presses (left or right button) and GUI interactions.
        """

        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()

            elif event.type == pg.MOUSEBUTTONDOWN and pg.mouse.get_pressed()[0]:
                x_pos, y_pos = pg.mouse.get_pos()
                if self.inside_canvas(x_pos, y_pos):
                    self.world[
                        :,
                        3:,
                        math.floor(y_pos / self.cell_size),
                        math.floor(x_pos / self.cell_size),
                    ] = 1

            elif event.type == pgui.UI_BUTTON_PRESSED:
                if event.ui_element == self.reset_button:
                    self.reset()

                elif event.ui_element == self.clear_button:
                    self.reset(spawn_seed=False)

                elif event.ui_element == self.pause_button:
                    self.paused = not self.paused
                    self.pause_button.set_text("Play" if self.paused else "Pause")

            elif event.type == pgui.UI_HORIZONTAL_SLIDER_MOVED:
                if event.ui_element == self.fps_slider:
                    self.fps = cast(int, self.fps_slider.get_current_value())
                    self.fps_label.set_text(f"FPS: {self.fps:>2d}")

                elif event.ui_element == self.angle_slider:
                    self.angle = self.angle_slider.get_current_value()
                    self.angle_label.set_text(f"Angle: {self.angle:>3.0f}??")

            elif event.type == pgui.UI_DROP_DOWN_MENU_CHANGED:
                if event.ui_element == self.model_name_selection:
                    self.model_type_selection.kill()

                    self.model_name = self.model_name_selection.selected_option.replace(
                        " ", ""
                    ).lower()
                    model = dutils.find_model(self.model_name, self.models)
                    self.model_type_selection = pgui.elements.UIDropDownMenu(
                        model.modeltypes,
                        model.modeltypes[0],
                        pg.Rect(self.model_type_pos[0], self.model_type_pos[1]),
                        self.ui_manager,
                    )
                    self.model_type = model.modeltypes[0].lower()

                    self.load_model()

                elif event.ui_element == self.model_type_selection:
                    self.model_type = self.model_type_selection.selected_option.replace(
                        " ", ""
                    ).lower()
                    self.load_model()

            self.ui_manager.process_events(event)

        if pg.mouse.get_pressed()[2]:
            x_pos, y_pos = pg.mouse.get_pos()
            if self.inside_canvas(x_pos, y_pos):
                x, y = math.floor(x_pos / self.cell_size), math.floor(y_pos / self.cell_size)
                self.world[:, :, y - 1 : y + 2, x - 1 : x + 2] = 0

    def inside_canvas(self, x_pos: int, y_pos: int) -> bool:
        """
        Checks whether a coordinate is inside the canvas (where the image is displayed).

        Args:
            x_pos (int): x position.
            y_pos (int): y position.

        Returns:
            bool: True, if coodinates are inside the canvas area. False, otherwise.
        """
        return x_pos < self.canvas_width and y_pos < self.canvas_height

    def step(self) -> npt.NDArray[np.int_]:
        """
        Performs one forward step of the model.

        Returns:
            npt.NDArray[np.int_]: Array of RGBA int values that are necesarry for drawing.
        """
        with torch.no_grad():
            self.world = self.model(self.world.to(self.device), angle=np.deg2rad(self.angle))
        colors = np.transpose(self.world[0, :4].cpu().numpy().clip(0, 1) * 255, (1, 2, 0)).astype(
            int
        )
        self.frame_count += 1
        return colors

    def draw_pixels(self, colors: npt.NDArray[np.int_]) -> None:
        """
        Draws all pixels of the currently generated image.

        Args:
            colors (npt.NDArray[np.int_]): Array of RGBA int values (e.g. the array returned by
            step())
        """
        for y in range(self.n_rows):
            for x in range(self.n_cols):
                r, g, b, a = colors[y, x]

                pixel = pg.Surface((self.cell_size, self.cell_size))
                pixel.set_alpha(a)
                pixel.fill((r, g, b))
                self.window.blit(pixel, (x * self.cell_size, y * self.cell_size))
                # TODO Do we need to perform alive masking here?
