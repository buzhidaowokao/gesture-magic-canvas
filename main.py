# -*- coding: utf-8 -*-
"""
《手势魔法画板》主程序 —— 摄像头实时手势识别互动作品（支持双手）

单手手势 → 互动效果：
    ☝ 单指（食指）  画笔作画
    ✌ 两指          画笔换黄色      三指 → 绿色    四指 → 蓝色
    🖐 张开手掌      彩虹粒子跟随指尖流动
    🤏 拇指食指捏合  绽放烟花
    ✊ 握拳 1 秒     清空画布（带进度环防误触）
    👍 点赞          保存作品到 saves/ 文件夹

双手配合 → 互动效果：
    🤏🤏 双手同时捏合  两指尖之间拉出一条流动的彩虹能量桥
    🖐🖐 双手同时张开  两手之间绽放大型多彩烟花
    （两只手各自独立识别，可以一只手画画、另一只手放烟花）

键盘快捷键：Q/ESC 退出 · S 保存 · C 清屏 · 1-5 换色

命令行参数：
    python main.py          使用 0 号摄像头启动
    python main.py 1        使用 1 号摄像头启动
    python main.py --fake   无摄像头演示模式（合成画面，用于自测）
    python main.py --smoke  自测模式：跑 90 帧后自动退出
"""

import os
os.environ.setdefault('GLOG_minloglevel', '3')  # 抑制 MediaPipe 启动日志

import sys
import time
import math
import random

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision

from gestures import MultiHandTracker
from particles import ParticleSystem, rainbow_color
from hud import HUD

_ROOT = os.path.dirname(os.path.abspath(__file__))
# 注意：mediapipe 的 C 库在 Windows 上无法打开含中文的绝对路径，
# 因此模型与保存路径都用相对路径，并在 main() 里 chdir 到项目根目录。
MODEL_PATH = os.path.join('models', 'hand_landmarker.task')
SAVE_DIR = os.path.join('saves')

# 注意：OpenCV 在 Windows 上不支持中文窗口标题（会显示乱码），
# 因此窗口标题用英文，中文标题由画面内的 HUD 呈现。
WINDOW_NAME = 'Gesture Magic Canvas - Q quit / S save / C clear'
CLEAR_HOLD_SECONDS = 1.0   # 握拳保持多久清屏
FIREWORK_COOLDOWN = 0.5    # 同一只手两次烟花最短间隔（秒）
BOOM_COOLDOWN = 1.0        # 双手掌大烟花最短间隔（秒）
SAVE_COOLDOWN = 2.0        # 两次保存最短间隔（秒）
STROKE_WIDTH = 6           # 笔迹粗细（像素）
SMOKE_FRAMES = 90          # --smoke 自测模式运行的帧数

# MediaPipe 手部骨骼连接关系（21 点两两相连）
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]
TIP_IDS = (4, 8, 12, 16, 20)  # 五个指尖

# 每种手势在界面上使用的强调色（BGR）
GESTURE_ACCENT = {
    'paint': (80, 80, 255), 'yellow': (30, 220, 255), 'green': (120, 220, 120),
    'blue': (255, 180, 80), 'rainbow': (255, 255, 255), 'firework': (80, 200, 255),
    'clear': (170, 170, 170), 'save': (60, 220, 60), 'none': (150, 150, 150),
}

# 画笔预设（数字键 1-5；'rainbow' 表示彩虹笔，颜色随时间循环）
BRUSH_PRESETS = {'1': (80, 80, 255), '2': (30, 220, 255),
                 '3': (120, 220, 120), '4': (255, 180, 80), '5': 'rainbow'}
BRUSH_NAMES = {'1': '红色', '2': '黄色', '3': '绿色', '4': '蓝色', '5': '彩虹'}


class FakeCamera:
    """无摄像头演示：生成合成画面，保证程序管线可完整运行"""

    def __init__(self, size=(1280, 720)):
        self.w, self.h = size
        self.n = 0

    def isOpened(self):
        return True

    def read(self):
        self.n += 1
        frame = np.zeros((self.h, self.w, 3), np.uint8)
        grad = np.linspace(30, 70, self.w, dtype=np.uint8)
        frame[:] = grad[None, :, None]
        cx = int(self.w / 2 + 380 * math.sin(self.n * 0.04))
        cy = int(self.h / 2 + 160 * math.cos(self.n * 0.03))
        cv2.circle(frame, (cx, cy), 70, (70, 160, 240), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 70, (255, 255, 255), 2, cv2.LINE_AA)
        return True, frame

    def release(self):
        pass


def draw_skeleton(frame, landmarks, accent):
    """绘制手部骨骼：粗的半透明底色 + 细的白色主线，指尖用强调色高亮"""
    pts = [(int(round(x)), int(round(y))) for x, y in landmarks]
    dim = tuple(int(c * 0.35) for c in accent)
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], dim, 7, cv2.LINE_AA)
        cv2.line(frame, pts[a], pts[b], (245, 245, 245), 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(pts):
        if i in TIP_IDS:
            cv2.circle(frame, (x, y), 7, accent, -1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (x, y), 3, (255, 255, 255), -1, cv2.LINE_AA)


def draw_bridge(frame, p1, p2, hue):
    """双手捏合的彩虹能量桥：两点之间一条流动的彩虹光束"""
    n = 24
    for i in range(n):
        t0, t1 = i / n, (i + 1) / n
        a = (int(p1[0] + (p2[0] - p1[0]) * t0), int(p1[1] + (p2[1] - p1[1]) * t0))
        b = (int(p1[0] + (p2[0] - p1[0]) * t1), int(p1[1] + (p2[1] - p1[1]) * t1))
        c = rainbow_color(hue + t0)
        dim = tuple(int(v * 0.35) for v in c)
        cv2.line(frame, a, b, dim, 9, cv2.LINE_AA)
        cv2.line(frame, a, b, c, 4, cv2.LINE_AA)
    for p in (p1, p2):  # 两端光球
        cv2.circle(frame, p, 10, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, p, 15, rainbow_color(hue), 2, cv2.LINE_AA)


class GestureMagicCanvas:
    """作品主体：摄像头采集 → 手势识别 → 互动逻辑 → 渲染显示"""

    def __init__(self, camera_index=0, fake=False, smoke=False):
        self.camera_index = camera_index
        self.fake = fake
        self.smoke = smoke

    # ---------- 初始化 ----------

    def _open_camera(self):
        if self.fake:
            return FakeCamera()
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)  # Windows 下 DSHOW 启动更快
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        if not cap.isOpened():  # 部分摄像头不支持 DSHOW，退回默认后端再试
            cap.release()
            cap = cv2.VideoCapture(self.camera_index)
        if cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                cap.release()
                cap = None
        return cap

    def _create_detector(self):
        """创建 MediaPipe 手部关键点检测器（VIDEO 模式，最多同时跟踪两只手）"""
        base = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.6,
        )
        return vision.HandLandmarker.create_from_options(options)

    def _save_snapshot(self, frame):
        """保存当前画面（含笔迹、手骨与特效）到 saves/ 文件夹

        用 imencode + Python 写文件，绕开 cv2.imwrite 在 Windows 上
        无法处理中文文件名的问题。"""
        os.makedirs(SAVE_DIR, exist_ok=True)
        name = time.strftime('作品_%Y%m%d_%H%M%S') + '.png'
        path = os.path.join(SAVE_DIR, name)
        ok, buf = cv2.imencode('.png', frame)
        if ok:
            with open(path, 'wb') as f:
                f.write(buf.tobytes())
        return path

    # ---------- 主循环 ----------

    def run(self):
        cap = self._open_camera()
        if cap is None:
            print(f'未检测到可用摄像头（索引 {self.camera_index}）。')
            print('请检查：1) 摄像头是否被其他软件占用；')
            print('        2) 若有多个摄像头，可运行 python main.py 1 切换。')
            return 1

        detector = self._create_detector()
        tracker = MultiHandTracker()   # 双手跟踪：每只手独立的平滑与防抖
        particles = ParticleSystem()
        hud = HUD()

        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)  # 镜像显示，符合照镜子直觉
        h, w = frame.shape[:2]
        canvas = np.zeros((h, w, 3), np.uint8)  # 画布层：只保存笔迹，与摄像头画面分离

        if not self.smoke:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, 960, 540)

        brush = BRUSH_PRESETS['1']   # 当前画笔（BGR 元组或 'rainbow'），两只手共用
        prev_tips = {}               # 每只手的画笔上一点 {hand_id: (x, y)}
        fire_last = {}               # 每只手的上次烟花时间 {hand_id: t}
        last_boom = 0.0              # 双手掌大烟花上次时间
        last_save = 0.0
        clear_armed = True           # 握拳清屏的一次性开关（防止持续握拳反复清）
        clear_hold = 0.0             # 已握拳累计时间
        white_flash = 0.0            # 保存成功时的白闪强度
        rainbow_hue = 0.0            # 彩虹色相位（随时间循环）
        fps = 0.0
        t_prev = None
        last_ts = 0                  # MediaPipe 要求时间戳严格递增
        save_requested = False       # 键盘 S 触发的保存（下一帧执行，保证截图不含 HUD）
        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print('摄像头画面读取中断，程序退出。')
                    break
                frame = cv2.flip(frame, 1)
                frame_count += 1

                t_now = time.monotonic()
                if t_prev is None:
                    dt = 1 / 30
                else:
                    dt = max(0.001, min(t_now - t_prev, 0.1))
                t_prev = t_now
                fps = fps * 0.9 + (1.0 / dt) * 0.1
                rainbow_hue = (rainbow_hue + dt * 0.35) % 1.0

                # ---------- 手势识别（最多两只手，各自独立跟踪） ----------
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ts = int(t_now * 1000)
                if ts <= last_ts:
                    ts = last_ts + 1
                last_ts = ts
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = detector.detect_for_video(mp_img, ts)

                # 组装 (关键点像素坐标, 左右手标签) 列表交给跟踪器
                dets = []
                for i, lms in enumerate(result.hand_landmarks):
                    handed = '?'
                    try:
                        handed = result.handedness[i][0].category_name
                    except (IndexError, AttributeError):
                        pass
                    dets.append(([(p.x * w, p.y * h) for p in lms], handed))
                tracked = tracker.update(dets, w)

                hand_infos = []   # 交给 HUD 显示的每只手信息
                current_ids = set()
                save_now = False

                # ---------- 每只手各自的持续/一次性动作 ----------
                for hid, hname, g, lm_sm in tracked:
                    current_ids.add(hid)
                    accent = GESTURE_ACCENT[g.name]
                    hand_infos.append(dict(handed=hname, key=g.name, label=g.label,
                                           desc=g.desc, accent=accent))

                    # —— 一次性动作：该手手势刚切换时触发 ——
                    if g.changed:
                        if g.name in ('yellow', 'green', 'blue'):
                            brush = BRUSH_PRESETS[{'yellow': '2', 'green': '3', 'blue': '4'}[g.name]]
                            hud.flash(f'{hname}换画笔：{g.label}', accent)
                        elif g.name == 'save' and t_now - last_save > SAVE_COOLDOWN:
                            save_now = True

                    # —— 持续动作（每只手独立状态，双手可同时画画） ——
                    if g.name == 'paint':
                        tip = (int(g.index_tip[0]), int(g.index_tip[1]))
                        color = rainbow_color(rainbow_hue) if brush == 'rainbow' else brush
                        prev = prev_tips.get(hid)
                        if prev is not None:
                            cv2.line(canvas, prev, tip, color, STROKE_WIDTH, cv2.LINE_AA)
                        else:  # 起笔画一个圆点
                            cv2.circle(canvas, tip, STROKE_WIDTH // 2, color, -1, cv2.LINE_AA)
                        prev_tips[hid] = tip
                    elif hid in prev_tips:
                        del prev_tips[hid]

                    if g.name == 'firework' and t_now - fire_last.get(hid, 0.0) > FIREWORK_COOLDOWN:
                        fire_last[hid] = t_now
                        particles.emit_firework(int(g.index_tip[0]), int(g.index_tip[1]))

                    if g.name == 'rainbow':
                        for k, tid in enumerate(TIP_IDS):
                            sx, sy = lm_sm[tid]
                            particles.emit_trail(int(sx), int(sy),
                                                 rainbow_color(rainbow_hue + k * 0.08), n=2)

                # 手已离开的，清掉它的画笔状态
                for hid in list(prev_tips):
                    if hid not in current_ids:
                        del prev_tips[hid]

                # —— 清屏：任一只手握拳即可，每帧只累计一次时间 ——
                clear_progress = 0.0
                clear_center = None
                fist = next((t for t in tracked if t[2].name == 'clear'), None)
                if fist is not None:
                    wrist = fist[3][0]
                    clear_center = (int(wrist[0]), int(wrist[1]))
                    if clear_armed:
                        clear_hold += dt
                        clear_progress = clear_hold / CLEAR_HOLD_SECONDS
                        if clear_hold >= CLEAR_HOLD_SECONDS:
                            canvas[:] = 0
                            particles.clear()
                            hud.flash('画布已清空', (170, 170, 170))
                            clear_armed = False  # 松开拳头后才允许再次清屏
                else:
                    clear_armed, clear_hold = True, 0.0

                # ---------- 合成显示：画布 → 粒子 → 手骨 ----------
                frame[:] = cv2.addWeighted(frame, 1.0, canvas, 0.9, 0)
                particles.update(dt)
                particles.draw(frame)
                for hid, hname, g, lm_sm in tracked:
                    draw_skeleton(frame, lm_sm, GESTURE_ACCENT[g.name])

                # ---------- 双手配合特效 ----------
                combo_key = None
                if len(tracked) == 2:
                    g1, g2 = tracked[0][2], tracked[1][2]
                    lm1, lm2 = tracked[0][3], tracked[1][3]
                    if g1.name == 'firework' and g2.name == 'firework':
                        # 双捏合 → 彩虹能量桥
                        combo_key = 'bridge'
                        p1 = (int(g1.index_tip[0]), int(g1.index_tip[1]))
                        p2 = (int(g2.index_tip[0]), int(g2.index_tip[1]))
                        draw_bridge(frame, p1, p2, rainbow_hue)
                        for _ in range(3):  # 光束沿途洒落粒子
                            t = random.random()
                            particles.emit_trail(
                                int(p1[0] + (p2[0] - p1[0]) * t),
                                int(p1[1] + (p2[1] - p1[1]) * t),
                                rainbow_color(rainbow_hue + random.random()), n=1)
                    elif g1.name == 'rainbow' and g2.name == 'rainbow':
                        # 双手掌 → 中央大烟花
                        combo_key = 'boom'
                        if t_now - last_boom > BOOM_COOLDOWN:
                            last_boom = t_now
                            mx = int((lm1[0][0] + lm2[0][0]) / 2)
                            my = int((lm1[0][1] + lm2[0][1]) / 2)
                            particles.emit_firework(mx, my, color=rainbow_color(rainbow_hue))
                            particles.emit_firework(mx, my, color=rainbow_color(rainbow_hue + 0.45))

                # ---------- 保存作品（在 HUD 绘制前截图，画面更干净） ----------
                if save_now or save_requested:
                    save_requested = False
                    if t_now - last_save > SAVE_COOLDOWN or save_now:
                        last_save = t_now
                        path = self._save_snapshot(frame)
                        hud.flash(f'已保存 {os.path.basename(path)}', (60, 220, 60), 2.2)
                        white_flash = 1.0

                if white_flash > 0:
                    a = 0.35 * white_flash
                    frame[:] = cv2.addWeighted(frame, 1 - a, np.full_like(frame, 255), a, 0)
                    white_flash = max(0.0, white_flash - dt * 2.5)

                # ---------- HUD 与窗口 ----------
                hud.draw(frame, hand_infos, fps, clear_progress, clear_center,
                         brush_bgr=rainbow_color(rainbow_hue) if brush == 'rainbow' else brush,
                         combo_key=combo_key)
                if not self.smoke:
                    cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1 if not self.smoke else 30) & 0xFF
                if key in (ord('q'), 27):
                    break
                elif key == ord('s'):
                    save_requested = True
                elif key == ord('c'):
                    canvas[:] = 0
                    particles.clear()
                    hud.flash('画布已清空', (170, 170, 170))
                elif key in (ord('1'), ord('2'), ord('3'), ord('4'), ord('5')):
                    ch = chr(key)
                    brush = BRUSH_PRESETS[ch]
                    hud.flash(f'画笔：{BRUSH_NAMES[ch]}',
                              brush if brush != 'rainbow' else (255, 255, 255))

                if self.smoke and frame_count >= SMOKE_FRAMES:
                    cv2.imwrite('smoke_preview.png', frame)  # 自测预览图，便于检查界面效果
                    print(f'--smoke 自测完成：已连续运行 {frame_count} 帧，'
                          f'最后一帧已存为 smoke_preview.png。')
                    break
        finally:
            detector.close()
            cap.release()
            cv2.destroyAllWindows()
        return 0


def main():
    os.chdir(_ROOT)  # 保证从任意目录启动时都能找到模型与保存目录
    args = sys.argv[1:]
    fake = '--fake' in args
    smoke = '--smoke' in args
    camera = int(next((a for a in args if a.isdigit()), 0))

    print('=' * 52)
    print('  《手势魔法画板》  Gesture Magic Canvas（双手版）')
    print('  把手举到镜头前试试：')
    print('    单指画画 / 两指黄 / 三指绿 / 四指蓝 / 手掌粒子')
    print('    捏合放烟花 / 握拳清屏 / 点赞保存')
    print('  双手玩法：双捏合拉彩虹桥 / 双手掌放大型烟花')
    print('  快捷键：Q 退出 | S 保存 | C 清屏 | 1-5 换色')
    print('=' * 52)
    sys.exit(GestureMagicCanvas(camera, fake=fake, smoke=smoke).run())


if __name__ == '__main__':
    main()
