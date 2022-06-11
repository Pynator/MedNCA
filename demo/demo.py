import math
import sys
from typing import cast

import numpy as np
import pygame as pg
import pygame_gui as pg_gui
import torch
from model.model import CAModel


class Demo:
    def __init__(self, start: bool = True) -> None:
        torch.no_grad()
        self.load_model("demo/trained_models/model")

        self.n_channels = 16
        self.n_cols = 28
        self.n_rows = 28

        self.world = torch.zeros(1, self.n_channels, self.n_cols, self.n_rows).to(self.device)
        # TODO Do we need to disable gradients or caching?

        self.init_pygame()
        if start:
            self.run()

    def load_model(self, path: str) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CAModel(n_channels=16, hidden_channels=128, fire_rate=0.5, device=self.device)
        self.model.load_state_dict(torch.load(path))

    def init_pygame(self) -> None:
        pg.init()

        # Window
        self.cell_size = 20

        self.image_width = self.n_cols * self.cell_size
        self.image_height = self.n_rows * self.cell_size
        self.sidebar_width = 400
        self.bottombar_height = 50
        self.width = self.image_width + self.sidebar_width
        self.height = self.image_height + self.bottombar_height

        self.window = pg.display.set_mode((self.width, self.height))
        pg.display.set_caption("Neural Cellular Automata")

        # Background
        self.background = pg.Surface((self.width, self.height))
        # self.background_color = pg.Color(0, 0, 0)
        self.background.fill((20, 20, 20))

        # Loop timing
        self.fps = 30
        self.clock = pg.time.Clock()

        # GUI
        self.ui_manager = pg_gui.UIManager((self.width, self.height))

        self.status_label = pg_gui.elements.UILabel(
            pg.Rect((0, self.image_height), (self.image_width, self.bottombar_height)),
            "",
            self.ui_manager,
        )
        self.clear_button = pg_gui.elements.UIButton(
            pg.Rect((self.image_width + 50, 50), (self.sidebar_width - 100, 50)),
            "Clear",
            self.ui_manager,
        )
        self.fps_slider = pg_gui.elements.UIHorizontalSlider(
            pg.Rect((self.image_width + 50, 150), (self.sidebar_width - 100, 50)),
            30,
            (5, 60),
            self.ui_manager,
            click_increment=5,
        )
        self.fps_label = pg_gui.elements.UILabel(
            pg.Rect((self.image_width + 50, 110), (self.sidebar_width - 100, 40)),
            "target fps: ...  |  current fps: ...",
            self.ui_manager,
        )

    def run(self) -> None:
        # Local variables
        cs = self.cell_size
        frame_count = 0

        # Main loop
        running = True
        while running:
            time_delta = self.clock.tick(self.fps) / 1000
            frame_count += 1

            # Handle events
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    sys.exit()

                elif event.type == pg.MOUSEBUTTONDOWN and pg.mouse.get_pressed()[0]:
                    x_pos, y_pos = pg.mouse.get_pos()
                    if self.inside_image(x_pos, y_pos):
                        self.world[:, 3:, math.floor(y_pos / cs), math.floor(x_pos / cs)] = 1

                elif event.type == pg_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == self.clear_button:
                        self.world = torch.zeros(1, self.n_channels, self.n_cols, self.n_rows)
                        frame_count = 1

                elif event.type == pg_gui.UI_HORIZONTAL_SLIDER_MOVED:
                    if event.ui_element == self.fps_slider:
                        self.fps = cast(int, self.fps_slider.get_current_value())

                self.ui_manager.process_events(event)

            if pg.mouse.get_pressed()[2]:
                x_pos, y_pos = pg.mouse.get_pos()
                if self.inside_image(x_pos, y_pos):
                    x, y = math.floor(x_pos / cs), math.floor(y_pos / cs)
                    self.world[:, :, y - 1 : y + 2, x - 1 : x + 2] = 0

            self.window.blit(self.background, (0, 0))

            # Update world
            self.world = self.model(self.world)
            colors = np.transpose(
                self.world[0, :4].detach().numpy().clip(0, 1) * 255, (1, 2, 0)
            ).astype(int)

            # Draw image
            for y in range(self.n_rows):
                for x in range(self.n_cols):
                    r, g, b, a = colors[y, x]

                    pixel = pg.Surface((cs, cs))
                    pixel.set_alpha(a)
                    pixel.fill((r, g, b))
                    self.window.blit(pixel, (x * cs, y * cs))
                    # TODO Do we need to perform alive masking here?

            # Draw image border
            pg.draw.rect(
                self.window, (200, 200, 200), (0, 0, self.n_cols * cs, self.n_rows * cs), width=2
            )
            # TODO Maybe display target image on sidebar

            # Print status
            self.status_label.set_text(f"step: {frame_count}")
            self.fps_label.set_text(
                f"target fps: {self.fps:>2d}  |  current fps: {self.clock.get_fps():>5.2f}"
            )

            self.ui_manager.update(time_delta)
            self.ui_manager.draw_ui(self.window)

            # Draw everthing to screen
            pg.display.flip()

    def inside_image(self, x_pos: int, y_pos: int) -> bool:
        return x_pos < self.image_width and y_pos < self.image_height
