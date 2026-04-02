"""
SkinRenderer — High-Quality 2D Visualization via Pygame + OpenGL
================================================================
Renders the skin condition environment state as a rich, real-time
medical dashboard:

  ┌──────────────────────────────────────────────────────┐
  │  SKIN CONDITION MONITOR            Day 12 / 30       │
  │  ┌──────────────┐  ┌────────────────────────────┐   │
  │  │  Skin Model  │  │   Severity Timeline         │   │
  │  │  (OpenGL     │  │   (sparkline chart)         │   │
  │  │   textured   │  └────────────────────────────┘   │
  │  │   face quad) │  ┌────────────────────────────┐   │
  │  └──────────────┘  │   Health Metrics Radar      │   │
  │  ┌─────────────────┤   (6-axis radar)             │   │
  │  │  Last Action   │  └────────────────────────────┘   │
  │  │  & Reward      │                                   │
  │  └─────────────────┘                                   │
  └──────────────────────────────────────────────────────┘

Dependencies: pygame, PyOpenGL, numpy
"""

import math
import numpy as np
import pygame
from pygame.locals import DOUBLEBUF, OPENGL, QUIT, KEYDOWN, K_ESCAPE
from OpenGL.GL import (
    glBegin, glEnd, glVertex2f, glVertex3f, glColor4f, glColor3f,
    glClear, glClearColor, glEnable, glDisable, glBlendFunc, glLineWidth,
    glMatrixMode, glLoadIdentity, glOrtho, glViewport,
    glPointSize, glFlush,
    glGenTextures, glBindTexture, glTexImage2D, glTexParameteri,
    glDeleteTextures, glTexCoord2f,
    GL_COLOR_BUFFER_BIT, GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
    GL_LINES, GL_LINE_STRIP, GL_LINE_LOOP, GL_TRIANGLES,
    GL_TRIANGLE_FAN, GL_TRIANGLE_STRIP, GL_QUADS, GL_POLYGON,
    GL_POINTS, GL_PROJECTION, GL_MODELVIEW, GL_DEPTH_TEST,
    GL_TEXTURE_2D, GL_RGBA, GL_UNSIGNED_BYTE,
    GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_LINEAR,
)
from OpenGL.GLU import gluOrtho2D
from environment.custom_env import ACTION_LABELS, ACTION_ICONS, MAX_DAYS


# ── palette ──────────────────────────────────────────────────────────────────
BG           = (0.06, 0.07, 0.10, 1.0)
PANEL        = (0.10, 0.12, 0.17, 1.0)
ACCENT_TEAL  = (0.18, 0.78, 0.68)
ACCENT_RED   = (0.92, 0.28, 0.35)
ACCENT_AMBER = (0.95, 0.72, 0.18)
ACCENT_BLUE  = (0.25, 0.55, 0.95)
TEXT_WHITE   = (0.95, 0.95, 0.98)
TEXT_DIM     = (0.50, 0.52, 0.58)
GRID_COLOR   = (0.15, 0.17, 0.22, 1.0)

SEVERITY_GRADIENT = [
    (0.20, 0.80, 0.45),   # 0.0 – healthy green
    (0.55, 0.80, 0.20),   # 0.25
    (0.95, 0.72, 0.18),   # 0.50 – amber warning
    (0.92, 0.45, 0.18),   # 0.75
    (0.92, 0.28, 0.35),   # 1.0 – critical red
]

RADAR_LABELS = ["Severity", "Inflam.", "Hydration", "Sun Dmg", "Stress", "Diet"]
RADAR_GOOD   = [False, False, True, False, False, True]   # True = higher is better


# ── utils ────────────────────────────────────────────────────────────────────

def lerp_color(colors, t):
    t = max(0.0, min(1.0, t))
    idx = t * (len(colors) - 1)
    lo  = int(idx)
    hi  = min(lo + 1, len(colors) - 1)
    frac = idx - lo
    c0, c1 = colors[lo], colors[hi]
    return tuple(c0[i] + (c1[i] - c0[i]) * frac for i in range(3))


def severity_color(v):
    return lerp_color(SEVERITY_GRADIENT, v)


def draw_rounded_rect(x, y, w, h, r=8, color=(1,1,1,1), segments=8):
    """Draw a filled rounded rectangle using triangle fan per corner."""
    glColor4f(*color)
    cx_list = [x+r, x+w-r, x+w-r, x+r]
    cy_list = [y+r, y+r,   y+h-r, y+h-r]
    angle_offsets = [180, 270, 0, 90]

    glBegin(GL_TRIANGLE_FAN)
    glVertex2f(x + w/2, y + h/2)
    for ci, (cx, cy, ao) in enumerate(zip(cx_list, cy_list, angle_offsets)):
        for s in range(segments + 1):
            a = math.radians(ao + s * 90 / segments)
            glVertex2f(cx + r * math.cos(a), cy + r * math.sin(a))
    glEnd()


def draw_text_pygame(surface, font, text, x, y, color=(240,240,248), anchor="left"):
    """Blit Pygame text onto a surface (used before OpenGL flip)."""
    rendered = font.render(text, True, color)
    if anchor == "center":
        x -= rendered.get_width() // 2
    elif anchor == "right":
        x -= rendered.get_width()
    surface.blit(rendered, (x, y))


# ── main renderer class ───────────────────────────────────────────────────────

class SkinRenderer:
    W, H = 1100, 680

    def __init__(self):
        pygame.init()
        self._screen = pygame.display.set_mode(
            (self.W, self.H), DOUBLEBUF | OPENGL
        )
        pygame.display.set_caption("DermRL — Skin Condition Monitor")

        # fonts
        pygame.font.init()
        self._font_title  = pygame.font.SysFont("DejaVuSans", 22, bold=True)
        self._font_body   = pygame.font.SysFont("DejaVuSans", 15)
        self._font_small  = pygame.font.SysFont("DejaVuSans", 12)
        self._font_icon   = pygame.font.SysFont("DejaVuSans", 20)
        self._font_big    = pygame.font.SysFont("DejaVuSans", 36, bold=True)

        # text overlay surface
        self._overlay = pygame.Surface((self.W, self.H), pygame.SRCALPHA)

        self._setup_gl()
        self._frame = 0

    def _setup_gl(self):
        glClearColor(*BG)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(0, self.W, 0, self.H)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    # ── public ───────────────────────────────────────────────────────────────

    def render(self, history: list, day: int, variant: int):
        """Main render call. Returns None (renders to window)."""
        for event in pygame.event.get():
            if event.type == QUIT:
                self.close()
                return
            if event.type == KEYDOWN and event.key == K_ESCAPE:
                self.close()
                return

        glClear(GL_COLOR_BUFFER_BIT)
        self._overlay.fill((0, 0, 0, 0))

        info = history[-1] if history else {}
        severity     = info.get("severity",     0.5)
        inflammation = info.get("inflammation", 0.5)
        hydration    = info.get("hydration",    0.5)
        sun_damage   = info.get("sun_damage",   0.3)
        stress       = info.get("stress",       0.5)
        diet_quality = info.get("diet_quality", 0.5)

        # ── layout regions ───────────────────────────────────────────────────
        # [face panel] left  x=20..350, y=80..440
        # [timeline]   right-top  x=370..740, y=80..280
        # [radar]      right-bot  x=370..740, y=300..520
        # [action log] far-right  x=760..1080, y=80..520
        # [header bar] top        x=0..W, y=640..680

        self._draw_header(day, variant)
        self._draw_face_panel(severity, inflammation, hydration)
        self._draw_timeline(history)
        self._draw_radar(severity, inflammation, hydration, sun_damage, stress, diet_quality)
        self._draw_action_log(history)
        self._draw_metric_bars(severity, inflammation, hydration, sun_damage, stress, diet_quality)

        # blit text overlay via pygame
        self._blit_overlay()

        pygame.display.flip()
        self._frame += 1
        pygame.time.wait(80)

    # ── sub-renderers ────────────────────────────────────────────────────────

    def _draw_header(self, day, variant):
        variant_names = ["Mild Acne", "Moderate Acne", "Severe Acne"]
        variant_colors = [(46, 200, 115), (242, 183, 47), (235, 71, 90)]
        # header background bar
        glColor4f(0.12, 0.14, 0.20, 1.0)
        glBegin(GL_QUADS)
        glVertex2f(0, self.H - 55); glVertex2f(self.W, self.H - 55)
        glVertex2f(self.W, self.H); glVertex2f(0, self.H)
        glEnd()

        draw_text_pygame(self._overlay, self._font_title,
                         "DermRL — Skin Condition Monitor",
                         24, 8, TEXT_WHITE)
        draw_text_pygame(self._overlay, self._font_title,
                         f"Day {day} / {MAX_DAYS}",
                         self.W - 160, 8, TEXT_WHITE)
        vc = variant_colors[variant % 3]
        draw_text_pygame(self._overlay, self._font_body,
                         f"● {variant_names[variant % 3]}",
                         24, 32, vc)

    def _draw_face_panel(self, severity, inflammation, hydration):
        """Draw animated procedural face with skin-tone texture."""
        px, py, pw, ph = 20, 70, 330, 400

        # panel bg
        draw_rounded_rect(px, self.H - py - ph, pw, ph,
                          r=12, color=(*PANEL[:3], 1.0))

        cx = px + pw / 2
        cy = self.H - py - ph / 2 - 20

        # skin tone based on severity
        skin_r = 0.88 - severity * 0.20
        skin_g = 0.72 - severity * 0.35
        skin_b = 0.60 - severity * 0.25

        # face oval
        glColor4f(skin_r, skin_g, skin_b, 1.0)
        segs = 64
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(cx, cy)
        for i in range(segs + 1):
            a = 2 * math.pi * i / segs
            glVertex2f(cx + 95 * math.cos(a), cy + 120 * math.sin(a))
        glEnd()

        # acne spots — more spots = higher severity
        n_spots = int(severity * 40)
        spot_rng = np.random.default_rng(42)   # deterministic per reset
        for _ in range(n_spots):
            sx = spot_rng.uniform(cx - 70, cx + 70)
            sy = spot_rng.uniform(cy - 90, cy + 80)
            # keep inside oval
            if ((sx - cx)/85)**2 + ((sy - cy)/110)**2 > 1.0:
                continue
            size  = spot_rng.uniform(2, 5 + inflammation * 5)
            alpha = spot_rng.uniform(0.5, 0.9)
            sc = severity_color(severity)
            glColor4f(*sc, alpha)
            glBegin(GL_TRIANGLE_FAN)
            glVertex2f(sx, sy)
            for i in range(13):
                a = 2 * math.pi * i / 12
                glVertex2f(sx + size * math.cos(a), sy + size * math.sin(a))
            glEnd()

        # inflammation glow overlay
        if inflammation > 0.3:
            glColor4f(0.90, 0.30, 0.20, inflammation * 0.25)
            glBegin(GL_TRIANGLE_FAN)
            glVertex2f(cx, cy)
            for i in range(segs + 1):
                a = 2 * math.pi * i / segs
                glVertex2f(cx + 95 * math.cos(a), cy + 120 * math.sin(a))
            glEnd()

        # eyes
        for ex in [cx - 32, cx + 32]:
            glColor4f(0.15, 0.12, 0.10, 1.0)
            glBegin(GL_TRIANGLE_FAN)
            glVertex2f(ex, cy + 25)
            for i in range(13):
                a = 2 * math.pi * i / 12
                glVertex2f(ex + 11 * math.cos(a), cy + 25 + 7 * math.sin(a))
            glEnd()
            # pupil
            glColor4f(0.05, 0.05, 0.08, 1.0)
            glBegin(GL_TRIANGLE_FAN)
            glVertex2f(ex, cy + 25)
            for i in range(9):
                a = 2 * math.pi * i / 8
                glVertex2f(ex + 4 * math.cos(a), cy + 25 + 4 * math.sin(a))
            glEnd()

        # nose
        glColor4f(skin_r - 0.08, skin_g - 0.06, skin_b - 0.05, 1.0)
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(cx, cy + 5)
        for i in range(9):
            a = 2 * math.pi * i / 8
            glVertex2f(cx + 8 * math.cos(a), cy + 5 + 10 * math.sin(a))
        glEnd()

        # mouth — frown/smile based on severity
        mouth_y = cy - 40
        curve = (1 - severity) * 12 - 6
        glColor4f(skin_r - 0.12, skin_g - 0.15, skin_b - 0.12, 1.0)
        glLineWidth(2.5)
        glBegin(GL_LINE_STRIP)
        for i in range(13):
            t = i / 12
            mx = cx - 30 + t * 60
            my = mouth_y + curve * math.sin(t * math.pi) * (-1 if severity > 0.5 else 1)
            glVertex2f(mx, my)
        glEnd()

        # hydration sheen
        if hydration > 0.6:
            glColor4f(0.6, 0.9, 1.0, (hydration - 0.6) * 0.3)
            glBegin(GL_TRIANGLE_FAN)
            glVertex2f(cx - 20, cy + 40)
            for i in range(9):
                a = 2 * math.pi * i / 8
                glVertex2f(cx - 20 + 15 * math.cos(a), cy + 40 + 20 * math.sin(a))
            glEnd()

        # label
        label = f"Severity: {severity:.0%}"
        col_rgb = tuple(int(c * 255) for c in severity_color(severity))
        draw_text_pygame(self._overlay, self._font_body, label,
                         px + pw // 2, self.H - py - ph + 10,
                         col_rgb, anchor="center")

    def _draw_timeline(self, history):
        """Severity sparkline chart."""
        rx, ry, rw, rh = 365, 70, 380, 190
        draw_rounded_rect(rx, self.H - ry - rh, rw, rh,
                          r=10, color=(*PANEL[:3], 1.0))

        draw_text_pygame(self._overlay, self._font_body,
                         "Severity Over Time", rx + 12, ry - 2, TEXT_DIM)

        if len(history) < 2:
            return

        data = [h["severity"] for h in history]
        mx = rx + 20;  my = self.H - ry - rh + 20
        gw = rw - 40;  gh = rh - 50

        # grid lines
        glColor4f(*GRID_COLOR)
        glLineWidth(1.0)
        for i in range(5):
            y = my + i * gh / 4
            glBegin(GL_LINES)
            glVertex2f(mx, y); glVertex2f(mx + gw, y)
            glEnd()

        # fill area under curve
        glBegin(GL_TRIANGLE_STRIP)
        for i, v in enumerate(data):
            t = i / max(len(data) - 1, 1)
            x = mx + t * gw
            y_top = my + (1 - v) * gh
            col = severity_color(v)
            glColor4f(*col, 0.35)
            glVertex2f(x, my)
            glColor4f(*col, 0.35)
            glVertex2f(x, y_top)
        glEnd()

        # line
        glLineWidth(2.5)
        glBegin(GL_LINE_STRIP)
        for i, v in enumerate(data):
            t = i / max(len(data) - 1, 1)
            x = mx + t * gw
            y = my + (1 - v) * gh
            glColor4f(*severity_color(v), 1.0)
            glVertex2f(x, y)
        glEnd()

        # current point
        v = data[-1]
        t = (len(data) - 1) / max(len(data) - 1, 1)
        x = mx + t * gw
        y = my + (1 - v) * gh
        glPointSize(8.0)
        glColor4f(*severity_color(v), 1.0)
        glBegin(GL_POINTS)
        glVertex2f(x, y)
        glEnd()

    def _draw_radar(self, severity, inflammation, hydration,
                    sun_damage, stress, diet_quality):
        """6-axis radar chart for health metrics."""
        rx, ry = 365, 290
        rw, rh  = 380, 240
        draw_rounded_rect(rx, self.H - ry - rh, rw, rh,
                          r=10, color=(*PANEL[:3], 1.0))

        draw_text_pygame(self._overlay, self._font_body,
                         "Health Metrics", rx + 12, ry - 2, TEXT_DIM)

        values_raw = [severity, inflammation, hydration, sun_damage, stress, diet_quality]
        # flip axes where higher = better
        values = []
        for i, v in enumerate(values_raw):
            values.append(1 - v if RADAR_GOOD[i] else v)

        cx = rx + rw / 2
        cy = self.H - ry - rh / 2 - 10
        R  = min(rw, rh) * 0.38
        n  = len(values)

        def pt(i, r):
            a = math.pi / 2 + 2 * math.pi * i / n
            return cx + r * math.cos(a), cy + r * math.sin(a)

        # web rings
        for ring in [0.25, 0.5, 0.75, 1.0]:
            glColor4f(0.20, 0.22, 0.30, 1.0)
            glLineWidth(1.0)
            glBegin(GL_LINE_LOOP)
            for i in range(n):
                x, y = pt(i, R * ring)
                glVertex2f(x, y)
            glEnd()

        # spokes
        glColor4f(0.20, 0.22, 0.30, 1.0)
        for i in range(n):
            x, y = pt(i, R)
            glBegin(GL_LINES)
            glVertex2f(cx, cy); glVertex2f(x, y)
            glEnd()

        # filled polygon
        glColor4f(*ACCENT_TEAL, 0.25)
        glBegin(GL_POLYGON)
        for i, v in enumerate(values):
            x, y = pt(i, R * v)
            glVertex2f(x, y)
        glEnd()

        glColor4f(*ACCENT_TEAL, 0.9)
        glLineWidth(2.0)
        glBegin(GL_LINE_LOOP)
        for i, v in enumerate(values):
            x, y = pt(i, R * v)
            glVertex2f(x, y)
        glEnd()

        # dots
        glPointSize(6.0)
        glBegin(GL_POINTS)
        for i, v in enumerate(values):
            col = severity_color(v)
            glColor4f(*col, 1.0)
            x, y = pt(i, R * v)
            glVertex2f(x, y)
        glEnd()

        # axis labels (text overlay)
        for i, label in enumerate(RADAR_LABELS):
            x, y = pt(i, R * 1.22)
            tx = int(x) - len(label) * 3
            ty = self.H - int(y) - 7
            draw_text_pygame(self._overlay, self._font_small, label,
                             tx, ty, (200, 210, 230))

    def _draw_action_log(self, history):
        """Right panel: last 8 actions with action name and severity delta."""
        rx, ry, rw, rh = 760, 70, 325, 460
        draw_rounded_rect(rx, self.H - ry - rh, rw, rh,
                          r=10, color=(*PANEL[:3], 1.0))
        draw_text_pygame(self._overlay, self._font_body,
                         "Action Log", rx + 12, ry - 2, TEXT_DIM)

        recent = history[-9:] if len(history) > 1 else []
        for idx, entry in enumerate(recent[1:]):   # skip initial state
            prev  = recent[idx]
            delta = prev["severity"] - entry["severity"]
            day   = entry["day"]
            sev   = entry["severity"]
            action_id = entry.get("action", -1)
            icon  = ACTION_ICONS[action_id]  if 0 <= action_id < len(ACTION_ICONS)  else "?"
            label = ACTION_LABELS[action_id] if 0 <= action_id < len(ACTION_LABELS) else "Unknown"

            col = (90, 210, 130) if delta > 0 else (220, 90, 100) if delta < -0.005 else (170, 175, 195)
            row_y = ry + 15 + idx * 48

            # row background
            bg_a = 0.08 if idx % 2 == 0 else 0.04
            draw_rounded_rect(rx + 8, self.H - row_y - 38, rw - 16, 36,
                              r=6, color=(0.18, 0.20, 0.28, bg_a))

            # day number
            draw_text_pygame(self._overlay, self._font_small,
                             f"Day {day:02d}", rx + 14, row_y + 2,
                             (150, 155, 175))

            # icon + action name (truncated to fit)
            short_label = label if len(label) <= 18 else label[:17] + "…"
            draw_text_pygame(self._overlay, self._font_small,
                             f"{icon} {short_label}", rx + 14, row_y + 16,
                             (210, 215, 230))

            # severity delta
            sign = "▲" if delta > 0 else ("▼" if delta < -0.005 else "–")
            draw_text_pygame(self._overlay, self._font_small,
                             f"{sign}{abs(delta):.3f}", rx + rw - 70, row_y + 16, col)

    def _draw_metric_bars(self, severity, inflammation, hydration,
                          sun_damage, stress, diet_quality):
        """Bottom strip under face panel: 6 metric bars."""
        values  = [severity, inflammation, hydration, sun_damage, stress, diet_quality]
        labels  = ["Severity", "Inflam.", "Hydration", "Sun Dmg", "Stress", "Diet"]
        good    = RADAR_GOOD

        bar_x = 20;   bar_start_y = self.H - 490
        bw    = 53;   bh = 100;  gap = 5

        for i, (v, lbl, is_good) in enumerate(zip(values, labels, good)):
            bx = bar_x + i * (bw + gap)
            by = bar_start_y

            # track bg
            draw_rounded_rect(bx, by, bw, bh, r=5, color=(0.12, 0.14, 0.20, 1.0))

            # filled portion
            fill = v if not is_good else v
            fill_h = int(bh * fill)
            fill_col = severity_color(1 - v if is_good else v)
            draw_rounded_rect(bx, by, bw, fill_h, r=5, color=(*fill_col, 0.9))

            # label
            draw_text_pygame(self._overlay, self._font_small, lbl,
                             bx + bw // 2, self.H - by - bh - 14,
                             (150, 155, 180), anchor="center")
            draw_text_pygame(self._overlay, self._font_small,
                             f"{v:.0%}",
                             bx + bw // 2, self.H - by - 14,
                             (220, 225, 235), anchor="center")

    def _blit_overlay(self):
        """Upload the pygame text surface as an OpenGL texture and draw it."""
        raw = pygame.image.tostring(self._overlay, "RGBA", True)
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.W, self.H, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, raw)
        glEnable(GL_TEXTURE_2D)

        glColor4f(1, 1, 1, 1)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(0, 0)
        glTexCoord2f(1, 0); glVertex2f(self.W, 0)
        glTexCoord2f(1, 1); glVertex2f(self.W, self.H)
        glTexCoord2f(0, 1); glVertex2f(0, self.H)
        glEnd()

        glDisable(GL_TEXTURE_2D)
        glDeleteTextures([tex])

    # ── cleanup ──────────────────────────────────────────────────────────────

    def close(self):
        pygame.quit()
