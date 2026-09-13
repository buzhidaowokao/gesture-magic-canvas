# -*- coding: utf-8 -*-
"""
hud.py —— 界面叠加层绘制

在摄像头画面上绘制：标题栏、当前手势徽章、右侧手势图例、
底部快捷键提示、清屏进度环、闪烁消息、无手引导动画。
中文文字通过 Pillow + 系统字体渲染（cv2.putText 不支持中文）。
"""

import math
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Windows 常见中文字体，按优先级尝试
_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyhbd.ttc",   # 微软雅黑 粗体
    r"C:\Windows\Fonts\msyh.ttc",     # 微软雅黑
    r"C:\Windows\Fonts\simhei.ttf",   # 黑体
    r"C:\Windows\Fonts\simsun.ttc",   # 宋体
]
_font_cache = {}


def _font(size, bold=False):
    """加载并缓存系统字体"""
    key = (size, bold)
    if key not in _font_cache:
        paths = ([_FONT_CANDIDATES[0]] if bold else []) + _FONT_CANDIDATES
        f = None
        for p in paths:
            try:
                f = ImageFont.truetype(p, size)
                break
            except OSError:
                continue
        _font_cache[key] = f or ImageFont.load_default()
    return _font_cache[key]


def _rgb(bgr):
    """OpenCV 的 BGR 元组 → Pillow 的 RGB 元组"""
    return (int(bgr[2]), int(bgr[1]), int(bgr[0]))


# 右侧图例：(手势代号, 图例文字, 圆点颜色 BGR)
LEGEND = [
    ('paint',    '单指 → 画笔作画',  (80, 80, 255)),
    ('yellow',   '两指 → 换黄色',    (30, 220, 255)),
    ('green',    '三指 → 换绿色',    (120, 220, 120)),
    ('blue',     '四指 → 换蓝色',    (255, 180, 80)),
    ('rainbow',  '手掌 → 彩虹粒子',  (255, 255, 255)),
    ('firework', '捏合 → 放烟花',    (80, 200, 255)),
    ('clear',    '握拳 → 清空画布',  (170, 170, 170)),
    ('save',     '点赞 → 保存作品',  (60, 220, 60)),
    ('bridge',   '双捏合 → 彩虹桥',  (255, 80, 255)),
    ('boom',     '双手掌 → 大烟花',  (0, 255, 255)),
]

_PANEL_FILL = (14, 16, 26, 185)      # 面板填充色（深蓝黑，半透明）
_PANEL_EDGE = (90, 110, 160, 255)    # 面板描边色
_TEXT_MAIN = (245, 247, 252, 255)    # 主文字
_TEXT_SUB = (200, 206, 220, 255)     # 次文字


class HUD:
    """界面绘制器：每帧调用 draw() 把所有 UI 画到画面上"""

    def __init__(self):
        self.messages = []  # 待显示的闪烁消息 [text, 过期时间, BGR颜色]

    def flash(self, text, bgr=(255, 255, 255), duration=1.8):
        """显示一条居中消息，duration 秒后淡出"""
        self.messages.append([text, time.monotonic() + duration, bgr])

    def draw(self, frame, hands, fps, clear_progress=0.0, clear_center=None,
             brush_bgr=(80, 80, 255), combo_key=None):
        """把全部 HUD 元素绘制到 frame 上（原地修改）

        hands: 每只手的信息列表，元素为 dict(handed, key, label, desc, accent)
        combo_key: 当前激活的双手配合特效代号（'bridge' / 'boom' / None）
        """
        h, w = frame.shape[:2]
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGBA')
        ov = Image.new('RGBA', img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)

        # ---- 标题栏 ----
        d.rounded_rectangle((12, 12, 316, 74), radius=12,
                            fill=_PANEL_FILL, outline=_PANEL_EDGE, width=1)
        d.text((28, 22), '手势魔法画板', font=_font(24, bold=True), fill=_TEXT_MAIN)
        d.text((28, 52), 'Gesture Magic Canvas', font=_font(12), fill=(150, 160, 180, 255))

        # ---- 当前手势徽章（每只手一行） ----
        panel_h = 36 + 46 * max(1, len(hands))
        by1 = 84 + panel_h
        d.rounded_rectangle((12, 84, 316, by1), radius=12,
                            fill=_PANEL_FILL, outline=_PANEL_EDGE, width=2)
        if hands:
            d.text((28, 92), f'手势 · 识别到 {len(hands)} 只手', font=_font(13),
                   fill=(150, 160, 180, 255))
            y = 84 + 32
            for info in hands:
                acc = _rgb(info['accent'])
                d.ellipse((28, y + 7, 42, y + 21), fill=acc + (255,))
                d.text((50, y), f"{info['handed']} · {info['label']}",
                       font=_font(18, bold=True), fill=acc + (255,))
                d.text((50, y + 23), info['desc'], font=_font(12), fill=_TEXT_SUB)
                y += 46
            # 画笔颜色指示圆点（面板右下角）
            d.ellipse((286, by1 - 26, 306, by1 - 6), fill=_rgb(brush_bgr) + (255,),
                      outline=(255, 255, 255, 220), width=1)
        else:
            d.text((28, 92), '手势 · 未识别', font=_font(13), fill=(150, 160, 180, 255))
            d.text((28, 116), '请将手掌举到镜头前', font=_font(16, bold=True),
                   fill=(220, 224, 235, 255))
            d.ellipse((286, 122, 306, 142), fill=_rgb(brush_bgr) + (255,),
                      outline=(255, 255, 255, 220), width=1)

        # ---- FPS 面板 ----
        d.rounded_rectangle((w - 132, 12, w - 12, 56), radius=10,
                            fill=_PANEL_FILL, outline=_PANEL_EDGE, width=1)
        d.text((w - 120, 24), f'{fps:4.1f} FPS', font=_font(16), fill=(180, 230, 180, 255))

        # ---- 手势图例 ----
        active_keys = {info['key'] for info in hands}
        lx0, ly0, row_h = w - 252, 84, 31
        lh = 40 + row_h * len(LEGEND) + 6
        d.rounded_rectangle((lx0, ly0, w - 12, ly0 + lh), radius=12,
                            fill=(14, 16, 26, 175), outline=_PANEL_EDGE, width=1)
        d.text((lx0 + 16, ly0 + 10), '手势图例', font=_font(17, bold=True),
               fill=(235, 238, 246, 255))
        for i, (key, text, dot) in enumerate(LEGEND):
            y = ly0 + 40 + i * row_h
            active = key in active_keys or key == combo_key
            if active:  # 当前手势所在行加亮背景
                d.rounded_rectangle((lx0 + 8, y - 3, w - 20, y + 23), radius=8,
                                    fill=(255, 255, 255, 36))
            d.ellipse((lx0 + 16, y + 3, lx0 + 30, y + 17), fill=_rgb(dot) + (255,))
            d.text((lx0 + 40, y - 1), text, font=_font(14),
                   fill=(255, 240, 170, 255) if active else (205, 212, 228, 255))

        # ---- 底部快捷键提示 ----
        d.rounded_rectangle((12, h - 42, w - 12, h - 12), radius=10,
                            fill=(14, 16, 26, 170))
        d.text((w // 2, h - 27), 'Q/ESC 退出 · S 保存 · C 清屏 · 1-5 换色',
               font=_font(14), fill=(190, 198, 216, 255), anchor='mm')

        # ---- 无手引导提示（呼吸闪烁） ----
        if not hands:
            pulse = 0.6 + 0.4 * math.sin(time.monotonic() * 3)
            d.text((w // 2, h // 2 - 40), '把一只手伸到镜头前试试吧',
                   font=_font(30, bold=True),
                   fill=(255, 255, 255, int(200 * pulse)), anchor='mm')

        # ---- 清屏进度环（画在手腕位置） ----
        if clear_progress > 0 and clear_center is not None:
            cx, cy = clear_center
            r = 52
            bbox = (cx - r, cy - r, cx + r, cy + r)
            d.ellipse(bbox, outline=(255, 255, 255, 70), width=6)
            p = min(clear_progress, 1.0)
            if p > 0.02:
                d.arc(bbox, start=-90, end=-90 + int(360 * p),
                      fill=(255, 120, 120, 255), width=6)
            d.text((cx, cy), f'{p * 100:.0f}%', font=_font(18, bold=True),
                   fill=(255, 255, 255, 230), anchor='mm')

        # ---- 闪烁消息（居中，淡出） ----
        now = time.monotonic()
        alive = []
        for i, (text, expire, bgr) in enumerate(self.messages):
            if now < expire:
                alive.append((text, expire, bgr))
                remain = expire - now
                alpha = 255 if remain > 0.5 else int(255 * remain / 0.5)
                tw = d.textlength(text, font=_font(18, bold=True))
                mx, my = w // 2, 110 + i * 46
                d.rounded_rectangle((mx - tw / 2 - 16, my - 20, mx + tw / 2 + 16, my + 20),
                                    radius=10, fill=(14, 16, 26, min(alpha, 200)))
                d.text((mx, my), text, font=_font(18, bold=True),
                       fill=_rgb(bgr) + (alpha,), anchor='mm')
        self.messages = alive

        # ---- 合成回 OpenCV 画面 ----
        out = Image.alpha_composite(img, ov).convert('RGB')
        frame[:] = cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2BGR)
