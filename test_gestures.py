# -*- coding: utf-8 -*-
"""
test_gestures.py —— 手势判定逻辑自测（无需摄像头）

用几何方法构造 8 种典型手势的 21 个关键点，喂给 GestureRecognizer，
验证分类结果与防抖逻辑是否符合预期。运行：python test_gestures.py
"""

from gestures import GestureRecognizer, STABLE_FRAMES


def make_hand(fingers=(1, 0, 0, 0), thumb='in', pinch=False, offset=(0, 0)):
    """
    构造一只朝上的手的 21 个关键点（归一化坐标 × 1000，当作像素）。

    fingers: 四指伸直状态 [食指, 中指, 无名指, 小指]，1=伸直 0=弯曲
    thumb:   'in' 收拢贴掌心 / 'out' 横向张开 / 'up' 竖起向上
    pinch:   True 时拇指尖与食指尖捏在一起
    offset:  整只手的平移量，用于构造两只分开的手
    """
    P = [None] * 21
    P[0] = (500, 850)          # 手腕
    P[9] = (500, 540)          # 中指根（手掌尺度 = 310）

    mcp = {8: (420, 560), 12: (500, 540), 16: (580, 560), 20: (650, 600)}
    P[5], P[13], P[17] = mcp[8], mcp[16], mcp[20]  # 其余三指的根部关键点
    ext_off = {   # 伸直时：PIP、TIP 相对 MCP 的偏移
        8: ((-5, -90), (-10, -180)), 12: ((0, -90), (0, -180)),
        16: ((5, -90), (10, -180)), 20: ((10, -80), (20, -150)),
    }
    fold_off = {  # 弯曲时：PIP、TIP 相对 MCP 的偏移（指尖卷回手掌）
        8: ((5, 40), (10, 80)), 12: ((0, 45), (0, 85)),
        16: ((-5, 40), (-10, 80)), 20: ((-5, 30), (-10, 60)),
    }

    for i, tip in enumerate((8, 12, 16, 20)):
        mx, my = mcp[tip]
        if pinch and tip == 8:
            # 食指弯向拇指，与拇指尖贴在一起
            P[6], P[7], P[8] = (430, 500), (418, 430), (412, 400)
            continue
        (pdx, pdy), (tdx, tdy) = ext_off[tip] if fingers[i] else fold_off[tip]
        P[tip - 2] = (mx + pdx, my + pdy)
        P[tip - 1] = (mx + pdx * 1.5, my + pdy * 1.5)
        P[tip] = (mx + tdx, my + tdy)

    if thumb == 'up':       # 拇指竖起
        P[1], P[2], P[3], P[4] = (440, 780), (400, 680), (340, 500), (280, 350)
    elif thumb == 'out':    # 拇指横向张开
        P[1], P[2], P[3], P[4] = (440, 750), (370, 700), (300, 660), (200, 600)
    elif thumb == 'pinchpose':
        P[1], P[2], P[3], P[4] = (440, 780), (400, 660), (405, 500), (412, 405)
    else:                   # 'in' 拇指收拢
        P[1], P[2], P[3], P[4] = (440, 780), (410, 700), (385, 640), (370, 600)

    if pinch:
        P[4] = (412, 398)   # 拇指尖贴到食指尖
    return [(float(x) + offset[0], float(y) + offset[1]) for x, y in P]


def hold(rec, landmarks, frames=STABLE_FRAMES):
    """连续喂入相同手型 frames 帧，返回最后一帧结果"""
    res = None
    for _ in range(frames):
        res = rec.update(landmarks)
    return res


def main():
    cases = [
        ('单指（画笔）',   make_hand((1, 0, 0, 0), 'in'),            'paint'),
        ('两指（换黄色）', make_hand((1, 1, 0, 0), 'in'),            'yellow'),
        ('三指（换绿色）', make_hand((1, 1, 1, 0), 'in'),            'green'),
        ('四指（换蓝色）', make_hand((1, 1, 1, 1), 'in'),            'blue'),
        ('手掌（粒子流）', make_hand((1, 1, 1, 1), 'out'),           'rainbow'),
        ('握拳（清屏）',   make_hand((0, 0, 0, 0), 'in'),            'clear'),
        ('点赞（保存）',   make_hand((0, 0, 0, 0), 'up'),            'save'),
        ('捏合（烟花）',   make_hand((1, 1, 1, 1), 'pinchpose', True), 'firework'),
        ('捏合·其余弯曲',  make_hand((0, 1, 1, 1), 'pinchpose', True), 'firework'),
    ]
    failed = 0
    for name, lm, expected in cases:
        rec = GestureRecognizer()
        res = hold(rec, lm)
        ok = res.name == expected
        failed += not ok
        mark = 'PASS' if ok else 'FAIL'
        print(f'[{mark}] {name:12s} 期望 {expected:9s} 实际 {res.name}')

    # ---- 防抖测试 ----
    rec = GestureRecognizer()
    lm = make_hand((1, 0, 0, 0), 'in')
    r1 = rec.update(lm)  # 只喂 1 帧
    ok1 = r1.name == 'none' and not r1.changed
    r2 = hold(rec, lm, STABLE_FRAMES - 1)  # 凑满 4 帧
    ok2 = r2.name == 'paint' and r2.changed
    print(f"[{'PASS' if ok1 else 'FAIL'}] 防抖：1 帧不生效（实际 {r1.name}）")
    print(f"[{'PASS' if ok2 else 'FAIL'}] 防抖：连续 {STABLE_FRAMES} 帧后生效并标记 changed（实际 {r2.name}）")
    failed += (not ok1) + (not ok2)

    # ---- 手势切换测试（关键点平滑需数帧收敛，多喂几帧） ----
    saw_change = False
    res = None
    for _ in range(STABLE_FRAMES + 4):
        res = rec.update(make_hand((1, 1, 1, 1), 'out'))
        saw_change = saw_change or res.changed
    ok3 = res.name == 'rainbow' and saw_change
    print(f"[{'PASS' if ok3 else 'FAIL'}] 切换：画笔 → 手掌（实际 {res.name}，触发切换事件 {saw_change}）")
    failed += not ok3

    # ---- 无手复位测试 ----
    rec.notify_no_hand()
    r4 = rec.update(make_hand((0, 0, 0, 0), 'in'))
    ok4 = r4.name == 'none' and not r4.changed
    print(f"[{'PASS' if ok4 else 'FAIL'}] 无手复位后需重新稳定（实际 {r4.name}）")
    failed += not ok4

    # ---- 抖动测试：交替手势不应误触发 ----
    rec = GestureRecognizer()
    hold(rec, make_hand((1, 1, 0, 0), 'in'))
    for _ in range(6):  # 交替抖动 6 次
        rec.update(make_hand((1, 0, 0, 0), 'in'))
        rec.update(make_hand((1, 1, 0, 0), 'in'))
    ok5 = rec._stable == 'yellow'
    print(f"[{'PASS' if ok5 else 'FAIL'}] 抖动：交替手型不误切换（实际 {rec._stable}）")
    failed += not ok5

    # ---- 双手跟踪测试 ----
    from gestures import MultiHandTracker
    tr = MultiHandTracker()
    left = make_hand((1, 0, 0, 0), 'in', offset=(-250, 0))    # 左侧：画笔手
    right = make_hand((1, 1, 1, 1), 'out', offset=(250, 0))   # 右侧：手掌手
    res = None
    for _ in range(STABLE_FRAMES + 2):
        res = tr.update([(left, 'Left'), (right, 'Right')], 1280)
    ok6 = (len(res) == 2
           and {t[2].name for t in res} == {'paint', 'rainbow'}
           and {t[1] for t in res} == {'左手', '右手'})
    print(f"[{'PASS' if ok6 else 'FAIL'}] 双手：两只手同时独立识别"
          f"（实际 {[t[2].name for t in res]}）")
    failed += not ok6

    ids = {t[0] for t in res}
    left2 = make_hand((1, 0, 0, 0), 'in', offset=(-240, 12))   # 两只手各微动一点
    right2 = make_hand((1, 1, 1, 1), 'out', offset=(240, -12))
    res2 = tr.update([(left2, 'Left'), (right2, 'Right')], 1280)
    ok7 = {t[0] for t in res2} == ids  # 微动后身份（id）应保持不变
    print(f"[{'PASS' if ok7 else 'FAIL'}] 双手：帧间微动后身份保持"
          f"（实际 {[t[0] for t in res2]}，期望 {sorted(ids)}）")
    failed += not ok7

    res3 = tr.update([(left2, 'Left')], 1280)  # 右手离开画面
    ok8 = len(res3) == 1 and res3[0][2].name == 'paint' and res3[0][0] in ids
    print(f"[{'PASS' if ok8 else 'FAIL'}] 双手：一只手离开后另一只不受影响"
          f"（实际 {[(t[0], t[2].name) for t in res3]}）")
    failed += not ok8

    res4 = tr.update([], 1280)  # 两只手都离开
    ok9 = len(res4) == 0
    print(f"[{'PASS' if ok9 else 'FAIL'}] 双手：全部离开后跟踪清空（实际 {len(res4)} 只）")
    failed += not ok9

    print('-' * 46)
    if failed:
        print(f'共 {failed} 项未通过！')
        raise SystemExit(1)
    print('全部测试通过 ✔')


if __name__ == '__main__':
    main()
