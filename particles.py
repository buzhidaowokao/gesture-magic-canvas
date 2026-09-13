# -*- coding: utf-8 -*-
"""
particles.py —— 粒子系统

负责两种特效：
1. 烟花：捏合手势触发，从指尖向四周爆发，带重力与亮度衰减；
2. 彩虹拖尾：张开手掌时，指尖持续洒出彩色光点。

实现方式：所有粒子先画到一张纯黑图层上，再用加法混合（cv2.add）
叠加到摄像头画面，亮度叠加饱和处会自然"过曝"发白，产生发光感。
"""

import math
import random
import colorsys

import cv2
import numpy as np

# 烟花随机色板（BGR）
FIREWORK_COLORS = [
    (60, 200, 255), (90, 170, 255), (120, 255, 160), (0, 255, 255),
    (30, 230, 255), (255, 200, 120), (200, 120, 255), (140, 255, 255),
]


def rainbow_color(t):
    """t ∈ [0,1) 循环 → 彩虹 BGR 颜色"""
    r, g, b = colorsys.hsv_to_rgb(t % 1.0, 1.0, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))


class Particle:
    __slots__ = ('x', 'y', 'vx', 'vy', 'life', 'max_life', 'size', 'color', 'gravity', 'drag')

    def __init__(self, x, y, vx, vy, life, size, color, gravity=0.0, drag=0.0):
        self.x, self.y = x, y          # 位置
        self.vx, self.vy = vx, vy      # 速度
        self.life = life               # 剩余寿命（秒）
        self.max_life = life           # 初始寿命（用于计算衰减比例）
        self.size = size               # 半径（像素）
        self.color = color             # BGR 颜色
        self.gravity = gravity         # 重力加速度（像素/秒²）
        self.drag = drag               # 空气阻力系数


class ParticleSystem:
    """粒子池：统一更新与绘制"""

    def __init__(self, max_particles=900):
        self.particles = []
        self.max_particles = max_particles

    def emit_firework(self, x, y, color=None):
        """在 (x, y) 绽放一朵烟花"""
        base = color if color is not None else random.choice(FIREWORK_COLORS)
        for _ in range(90):
            ang = random.uniform(0, math.tau)
            spd = random.uniform(50, 430) * (random.random() ** 0.5)  # 外密内疏
            life = random.uniform(0.7, 1.4)
            size = random.uniform(1.5, 3.5)
            jitter = random.uniform(0.85, 1.0)  # 每颗粒子颜色略作明暗抖动
            c = tuple(int(ch * jitter) for ch in base)
            self.particles.append(Particle(
                x, y, math.cos(ang) * spd, math.sin(ang) * spd,
                life, size, c, gravity=260, drag=1.6))
        # 中心白色闪光
        self.particles.append(Particle(x, y, 0, 0, 0.25, 14, (255, 255, 255)))

    def emit_trail(self, x, y, color, n=2):
        """在 (x, y) 洒出 n 颗短寿命拖尾光点"""
        for _ in range(n):
            self.particles.append(Particle(
                x + random.uniform(-4, 4), y + random.uniform(-4, 4),
                random.uniform(-25, 25), random.uniform(-35, 5),
                random.uniform(0.35, 0.7), random.uniform(1.2, 2.6),
                color, gravity=18, drag=1.2))

    def clear(self):
        self.particles.clear()

    def update(self, dt):
        """推进一帧物理：速度受阻力衰减、叠加重力，寿命递减"""
        alive = []
        for p in self.particles:
            p.life -= dt
            if p.life <= 0:
                continue
            k = max(0.0, 1.0 - p.drag * dt)
            p.vx *= k
            p.vy = p.vy * k + p.gravity * dt
            p.x += p.vx * dt
            p.y += p.vy * dt
            alive.append(p)
        self.particles = alive
        if len(self.particles) > self.max_particles:
            del self.particles[:len(self.particles) - self.max_particles]

    def draw(self, frame):
        """把粒子加法混合叠加到画面上（原地修改）"""
        if not self.particles:
            return frame
        layer = np.zeros_like(frame)
        for p in self.particles:
            t = p.life / p.max_life
            r = max(1, int(round(p.size * (0.4 + 0.6 * t))))
            c = tuple(int(ch * t) for ch in p.color)  # 寿命越少越暗
            cv2.circle(layer, (int(p.x), int(p.y)), r, c, -1, cv2.LINE_AA)
        frame[:] = cv2.add(frame, layer)
        return frame
