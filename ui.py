"""
ui.py - colours, fonts, text helpers and a Button, shared by every screen
(gui.py for settings/game, tree_view.py for the search-tree viewer).
"""

import pygame

from board import HUMAN, AI

FPS = 60
BG = (22, 26, 38)
PANEL = (32, 38, 54)
BOARD_BLUE = (32, 96, 210)
BOARD_EDGE = (22, 66, 160)
HOLE_BG = (14, 17, 26)
HUMAN_COLOR = (222, 52, 52)
AI_COLOR = (246, 200, 40)
TEXT = (236, 239, 246)
MUTED = (150, 158, 178)
BTN = (54, 63, 88)
BTN_HOVER = (72, 84, 116)
BTN_ON = (48, 120, 222)
BTN_DISABLED = (40, 45, 60)

PIECE_COLORS = {HUMAN: HUMAN_COLOR, AI: AI_COLOR}

_font_cache = {}


def font(size, bold=False):
    """Cached system font (the tree viewer asks for many sizes when zooming)."""
    key = (int(size), bold)
    if key not in _font_cache:
        _font_cache[key] = pygame.font.SysFont("segoeui,arial,helvetica",
                                               max(6, int(size)), bold=bold)
    return _font_cache[key]


def draw_text(surf, text, fnt, color, pos, anchor="topleft"):
    img = fnt.render(text, True, color)
    rect = img.get_rect(**{anchor: pos})
    surf.blit(img, rect)
    return rect


def wrap_text(text, fnt, width):
    """Split `text` into lines that fit inside `width` pixels."""
    words, lines, line = text.split(), [], ""
    for w in words:
        test = (line + " " + w).strip()
        if fnt.size(test)[0] <= width:
            line = test
        else:
            if line:
                lines.append(line)
            line = w
    if line:
        lines.append(line)
    return lines


class Button:
    """A minimal clickable button. `selected` / `enabled` are callables so
    the button always reflects the current state."""

    def __init__(self, rect, label, on_click,
                 selected=lambda: False, enabled=lambda: True, size=20):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.on_click = on_click
        self.selected = selected
        self.enabled = enabled
        self.size = size

    def draw(self, surf, mouse):
        if not self.enabled():
            color, txt = BTN_DISABLED, MUTED
        elif self.selected():
            color, txt = BTN_ON, TEXT
        elif self.rect.collidepoint(mouse):
            color, txt = BTN_HOVER, TEXT
        else:
            color, txt = BTN, TEXT
        pygame.draw.rect(surf, color, self.rect, border_radius=8)
        draw_text(surf, self.label, font(self.size, True), txt,
                  self.rect.center, "center")

    def handle(self, event):
        if (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                and self.rect.collidepoint(event.pos) and self.enabled()):
            self.on_click()
            return True
        return False
