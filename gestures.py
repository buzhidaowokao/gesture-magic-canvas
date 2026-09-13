# -*- coding: utf-8 -*-
"""
gestures.py —— 手势识别模块

基于 MediaPipe HandLandmarker 输出的 21 个手部关键点，
使用纯几何规则判定当前手势（无需额外模型，全部离线完成）。

判定思路：
1. 用「手腕 → 中指根部」的距离作为手掌尺度，使所有阈值与镜头远近无关；
2. 四指：手腕到指尖的距离 > 手腕到第二关节(PIP)的距离 × 系数 → 伸直；
3. 拇指：拇指尖远离小指根或手腕 → 伸展；
4. 捏合：拇指尖与食指尖距离小于手掌尺度 × 系数；
5. 综合上述特征与伸出手指数量，按优先级映射为 8 种手势；
6. 连续多帧相同才生效（防抖），关键点做指数平滑（防手抖锯齿）。
"""

import math
from collections import deque

# ---------- MediaPipe 21 个手部关键点索引 ----------
WRIST = 0          # 手腕
THUMB_TIP = 4      # 拇指尖
INDEX_TIP = 8      # 食指尖
MIDDLE_TIP = 12    # 中指尖
RING_TIP = 16      # 无名指尖
PINKY_TIP = 20     # 小指尖
INDEX_PIP = 6      # 食指第二关节
MIDDLE_PIP = 10    # 中指第二关节
RING_PIP = 14      # 无名指第二关节
PINKY_PIP = 18     # 小指第二关节
MIDDLE_MCP = 9     # 中指根部（用于计算手掌尺度）
PINKY_MCP = 17     # 小指根部（用于拇指判定）

# ---------- 可调参数（若识别过于灵敏/迟钝，可微调这里） ----------
FINGER_EXT_RATIO = 1.12    # 手指伸直判定系数
THUMB_SPREAD_RATIO = 1.35  # 拇指伸直判定：拇指尖—小指根距离 / 手掌尺度
THUMB_AWAY_RATIO = 1.45    # 拇指伸直备选：拇指尖—手腕距离 / 手掌尺度
PINCH_RATIO = 0.45         # 捏合判定：拇指尖—食指尖距离 / 手掌尺度
STABLE_FRAMES = 4          # 防抖帧数：连续 N 帧相同手势才生效
SMOOTH_ALPHA = 0.5         # 关键点平滑系数（0~1，越小越平滑但延迟越大）

# ---------- 手势代号 → 界面显示（中文名, 动作说明） ----------
GESTURE_LABELS = {
    'paint':    ('画笔',   '移动食指即可作画'),
    'yellow':   ('黄色',   '画笔换成黄色'),
    'green':    ('绿色',   '画笔换成绿色'),
    'blue':     ('蓝色',   '画笔换成蓝色'),
    'rainbow':  ('粒子流', '彩虹粒子跟随指尖'),
    'firework': ('烟花',   '绽放一朵烟花！'),
    'clear':    ('清屏',   '保持握拳 1 秒清空画布'),
    'save':     ('保存',   '保存当前作品'),
    'none':     ('未识别', '请将手掌举到镜头前'),
}


def dist(a, b):
    """两点间欧氏距离"""
    return math.hypot(a[0] - b[0], a[1] - b[1])


class GestureResult:
    """一帧的识别结果"""

    __slots__ = ('name', 'label', 'desc', 'changed', 'fingers',
                 'index_tip', 'thumb_tip', 'pinch_dist')

    def __init__(self, name, changed, fingers, index_tip, thumb_tip, pinch_dist):
        self.name = name              # 稳定后的手势代号（见 GESTURE_LABELS）
        self.label, self.desc = GESTURE_LABELS[name]
        self.changed = changed        # 手势是否刚发生切换（用于触发一次性动作）
        self.fingers = fingers        # 五指伸展状态 [拇指, 食指, 中指, 无名指, 小指]
        self.index_tip = index_tip    # 平滑后的食指尖像素坐标 (x, y)
        self.thumb_tip = thumb_tip    # 平滑后的拇指尖像素坐标
        self.pinch_dist = pinch_dist  # 拇指尖—食指尖距离（像素）


class GestureRecognizer:
    """手势识别器：喂入像素坐标关键点，输出稳定后的手势"""

    def __init__(self, stable_frames=STABLE_FRAMES):
        self.stable_frames = stable_frames
        self._history = deque(maxlen=stable_frames)
        self._stable = 'none'
        self._smoothed = None  # 平滑后的 21 个关键点

    @property
    def smoothed_landmarks(self):
        return self._smoothed

    def update(self, landmarks):
        """landmarks: 21 个 (x, y) 像素坐标列表（应与显示画面同向，即先镜像）"""
        # 关键点指数平滑，抑制手抖带来的锯齿
        if self._smoothed is None:
            self._smoothed = [(float(x), float(y)) for x, y in landmarks]
        else:
            a = SMOOTH_ALPHA
            self._smoothed = [
                (a * x + (1 - a) * ox, a * y + (1 - a) * oy)
                for (x, y), (ox, oy) in zip(landmarks, self._smoothed)
            ]
        lm = self._smoothed
        fingers = self._fingers_of(lm)
        raw = self._classify(lm, fingers)
        self._history.append(raw)
        changed = False
        if (len(self._history) == self.stable_frames
                and len(set(self._history)) == 1
                and raw != self._stable):
            self._stable = raw
            changed = True
        return GestureResult(self._stable, changed, fingers,
                             lm[INDEX_TIP], lm[THUMB_TIP],
                             dist(lm[THUMB_TIP], lm[INDEX_TIP]))

    def notify_no_hand(self):
        """画面中没有手时调用：清空历史，避免旧手势残留"""
        self._history.clear()
        self._stable = 'none'
        self._smoothed = None

    # ---------- 内部实现 ----------

    def _fingers_of(self, lm):
        """判定五指伸展状态，返回 [拇指, 食指, 中指, 无名指, 小指] 的布尔列表"""
        thumb = self._thumb_extended(lm)
        others = [
            dist(lm[WRIST], lm[tip]) > dist(lm[WRIST], lm[pip]) * FINGER_EXT_RATIO
            for tip, pip in ((INDEX_TIP, INDEX_PIP), (MIDDLE_TIP, MIDDLE_PIP),
                             (RING_TIP, RING_PIP), (PINKY_TIP, PINKY_PIP))
        ]
        return [thumb] + others

    def _thumb_extended(self, lm):
        """拇指判定：指尖明显远离小指根（横向张开）或远离手腕（竖起）"""
        scale = dist(lm[WRIST], lm[MIDDLE_MCP])
        spread = dist(lm[THUMB_TIP], lm[PINKY_MCP])
        away = dist(lm[THUMB_TIP], lm[WRIST])
        return spread > scale * THUMB_SPREAD_RATIO or away > scale * THUMB_AWAY_RATIO

    def _classify(self, lm, fingers):
        """核心规则：按优先级把手部特征映射为手势代号"""
        scale = dist(lm[WRIST], lm[MIDDLE_MCP])
        if scale < 1e-6:
            return 'none'
        count = sum(fingers)
        pinch_dist = dist(lm[THUMB_TIP], lm[INDEX_TIP])
        # 1) 捏合 → 烟花（要求中指伸直，避免与握拳混淆）
        if pinch_dist < scale * PINCH_RATIO and fingers[2]:
            return 'firework'
        # 2) 五指全收 → 握拳清屏
        if count == 0:
            return 'clear'
        # 3) 只伸拇指且拇指尖远离食指尖 → 点赞保存
        if count == 1 and fingers[0] and pinch_dist > scale:
            return 'save'
        # 4) 只伸食指 → 画笔
        if count == 1 and fingers[1]:
            return 'paint'
        # 5) 按伸出手指数量映射颜色与特效
        return {5: 'rainbow', 4: 'blue', 3: 'green', 2: 'yellow'}.get(count, 'none')


# ==================== 双手跟踪 ====================

def handed_label(name):
    """MediaPipe 的左右手标签 → 中文（画面已镜像，结果即真实的左右手）"""
    return {'Left': '左手', 'Right': '右手'}.get(name, '手')


class HandSlot:
    """一只被持续跟踪的手：拥有独立的关键点平滑、防抖历史与画笔状态"""

    __slots__ = ('hand_id', 'recognizer', 'anchor', 'handed', 'matched')

    def __init__(self, hand_id):
        self.hand_id = hand_id
        self.recognizer = GestureRecognizer()
        self.anchor = None   # 上一帧手腕位置，用于帧间身份匹配
        self.handed = '?'
        self.matched = False


class MultiHandTracker:
    """双手跟踪器

    MediaPipe 每帧输出的手没有稳定编号（两只手在列表里的顺序可能互换），
    这里用「手腕位置最近邻匹配」把前后帧的手一一对应起来，
    让每只手始终使用自己的平滑与防抖状态，两只手互不干扰。
    """

    def __init__(self, match_dist_ratio=0.35):
        # 匹配距离上限 = 画面宽度 × 该系数（手在相邻两帧间一般不会瞬移这么远）
        self.match_dist_ratio = match_dist_ratio
        self.slots = []
        self._next_id = 0

    def update(self, detections, frame_w):
        """
        detections: [(21点像素坐标, 左右手标签字符串), ...]
        返回: [(hand_id, 中文左右手, GestureResult, 平滑后关键点), ...]
        """
        limit = frame_w * self.match_dist_ratio

        # 贪心最近邻：所有(旧手, 新手)配对按距离从小到大依次锁定
        pairs = []
        for si, slot in enumerate(self.slots):
            if slot.anchor is None:
                continue
            for di, (lm, _) in enumerate(detections):
                d = dist(lm[WRIST], slot.anchor)
                if d <= limit:
                    pairs.append((d, si, di))
        pairs.sort(key=lambda p: p[0])

        used_s, used_d, det_slot = set(), set(), {}
        for _, si, di in pairs:
            if si in used_s or di in used_d:
                continue
            used_s.add(si)
            used_d.add(di)
            det_slot[di] = self.slots[si]

        # 逐只手更新状态；没匹配上旧手的新手 → 新建槽位
        results, alive = [], []
        for di, (lm, handed) in enumerate(detections):
            slot = det_slot.get(di)
            if slot is None:
                slot = HandSlot(self._next_id)
                self._next_id += 1
            slot.matched = True
            slot.anchor = (lm[WRIST][0], lm[WRIST][1])
            slot.handed = handed
            res = slot.recognizer.update(lm)
            alive.append(slot)
            results.append((slot.hand_id, handed_label(handed),
                            res, slot.recognizer.smoothed_landmarks))

        # 上一帧存在、这一帧消失的手 → 复位并移除
        for slot in self.slots:
            if not slot.matched:
                slot.recognizer.notify_no_hand()
        self.slots = alive
        for slot in self.slots:
            slot.matched = False
        return results
