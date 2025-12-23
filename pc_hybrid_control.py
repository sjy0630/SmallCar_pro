import sounddevice as sd
import numpy as np
import requests
import json
import time
import os
import sys
import cv2
import torch
import re
from threading import Thread, Lock

# --- 引入依赖 ---
from funasr import AutoModel
from sherpa_onnx import VadModelConfig, SileroVadModelConfig, VoiceActivityDetector
import model.detector
import utils.utils

# ==========================================
#               全局配置区
# ==========================================

CAR_IP = "172.24.225.17"
CONTROL_URL = f"http://{CAR_IP}:5000/control"
STREAM_URL = f"http://{CAR_IP}:8080/?action=stream"

API_KEY = "sk-8259e96168f94f4baf816ec8769b726a"
API_URL = "https://api.deepseek.com/chat/completions"
LLM_MODEL_NAME = "deepseek-chat"

VAD_PATH = 'voice_models/VAD/silero_vad.onnx' 
ASR_MODEL_PATH = "iic/SenseVoiceSmall" 
YOLO_CFG_DATA = 'data/coco.data'
YOLO_WEIGHTS = 'modelzoo/coco2017-0.241078ap-model.pth'
YOLO_NAMES = 'data/coco.names'

# 映射表
CN_COCO_MAP = {
    "人": "person", "我": "person", "自己": "person",
    "瓶": "bottle", "水": "bottle",
    "杯": "cup",
    "手机": "cell phone", "电话": "cell phone",
    "书": "book",
    "猫": "cat", "狗": "dog",
    "球": "sports ball",
    "车": "car", "椅": "chair", "键盘": "keyboard", "鼠标": "mouse"
}

# 全局状态
CURRENT_MODE = "VOICE"
CURRENT_TARGET = "person"
PROGRAM_RUNNING = True
# 新增：强制解锁信号
FORCE_UNLOCK_SIGNAL = False

video_lock = Lock()
latest_frame = None
latest_ret = False

session = requests.Session()
session.headers.update({"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"})

# ==========================================
#           模块 1: 视觉追踪 (v3 增强版)
# ==========================================

Kp = 0.35; Ki = 0.0; Kd = 1.0
CENTER_DEAD_ZONE = 0.12
STOP_AREA_THRESHOLD = 0.3 # 稍微放宽停止距离
MAX_LOCK_AREA = 0.6       # 放宽最大面积，防止跟丢近处的物体
# [关键修改] 大幅降低置信度阈值，让它能看到更多东西
CONF_THRESHOLD = 0.20     
IOU_THRESHOLD = 0.4
ERROR_BUFFER_SIZE = 3

integral = 0.0
previous_error = 0.0
last_pid_time = time.time()
error_buffer = []

def load_yolo_model():
    print("📷 加载 YOLO 模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = utils.utils.load_datafile(YOLO_CFG_DATA)
    yolo_net = model.detector.Detector(cfg["classes"], cfg["anchor_num"], True).to(device)
    yolo_net.load_state_dict(torch.load(YOLO_WEIGHTS, map_location=device))
    yolo_net.eval()
    
    label_names = ["person"]
    if os.path.exists(YOLO_NAMES):
        with open(YOLO_NAMES, 'r', encoding='utf-8') as f:
            label_names = [line.strip() for line in f.readlines()]
    return yolo_net, device, cfg, label_names

def get_pid_command(box, h, w):
    global integral, previous_error, last_pid_time, error_buffer
    
    x_center = (box[0] + box[2]) / 2 / w
    box_area = (box[2] - box[0]) * (box[3] - box[1]) / (h * w)
    
    if box_area > STOP_AREA_THRESHOLD:
        integral = 0.0; previous_error = 0.0; error_buffer = []
        return "STOP", 0.0
    
    curr_time = time.time()
    dt = curr_time - last_pid_time
    if dt == 0: dt = 1e-5
    
    target_x = 0.5
    raw_error = x_center - target_x
    
    error_buffer.append(raw_error)
    if len(error_buffer) > ERROR_BUFFER_SIZE: error_buffer.pop(0)
    smooth_error = sum(error_buffer) / len(error_buffer)
    
    if abs(smooth_error) < CENTER_DEAD_ZONE:
        smooth_error = 0.0; previous_error = 0.0
        
    integral += smooth_error * dt
    integral = max(-1.0, min(1.0, integral))
    derivative = (smooth_error - previous_error) / dt
    
    # 针对小物体增加灵敏度
    active_Kp = 0.55 if CURRENT_TARGET != "person" else 0.35
    steer = (active_Kp * smooth_error) + (Ki * integral) + (Kd * derivative)
    previous_error = smooth_error
    last_pid_time = curr_time
    
    return "FORWARD", max(-1.0, min(1.0, steer))

def video_loop():
    global latest_frame, latest_ret, CURRENT_MODE, PROGRAM_RUNNING, CURRENT_TARGET, FORCE_UNLOCK_SIGNAL
    
    yolo_net, device, cfg, label_names = load_yolo_model()
    
    cap = cv2.VideoCapture(STREAM_URL)
    
    def read_stream():
        global latest_frame, latest_ret
        while PROGRAM_RUNNING:
            ret, frame = cap.read()
            with video_lock:
                latest_frame = frame; latest_ret = ret
            if not ret: time.sleep(0.5)
            else: time.sleep(0.01)
    
    t_read = Thread(target=read_stream, daemon=True)
    t_read.start()
    
    tracker = None
    is_tracking = False
    frames_since_detect = 0
    RE_DETECT_INTERVAL = 10 # 加快重检测频率
    
    print("✅ 视觉线程启动")
    
    while PROGRAM_RUNNING:
        with video_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
            ret = latest_ret
            
        if not ret or frame is None:
            time.sleep(0.1); continue
            
        h, w, _ = frame.shape
        scale_h, scale_w = h / cfg["height"], w / cfg["width"]
        
        # --- 处理强制解锁信号 ---
        if FORCE_UNLOCK_SIGNAL:
            print("🔓 视觉收到指令: 解除锁定")
            is_tracking = False
            tracker = None
            FORCE_UNLOCK_SIGNAL = False # 复位信号
            send_command("STOP", 0.0, 0.0)

        # 仅在 TRACK 模式工作
        if CURRENT_MODE == "TRACK":
            need_detection = (not is_tracking) or (frames_since_detect > RE_DETECT_INTERVAL)
            yolo_boxes = []
            
            if need_detection:
                frames_since_detect = 0
                res_img = cv2.resize(frame, (cfg["width"], cfg["height"]), interpolation=cv2.INTER_LINEAR)
                img = res_img.reshape(1, cfg["height"], cfg["width"], 3)
                img = torch.from_numpy(img.transpose(0, 3, 1, 2)).to(device).float() / 255.0
                
                preds = yolo_net(img)
                output = utils.utils.handel_preds(preds, cfg, device)
                output_boxes = utils.utils.non_max_suppression(output, conf_thres=CONF_THRESHOLD, iou_thres=IOU_THRESHOLD)
                
                for box in output_boxes[0]:
                    box_list = box.tolist()
                    cls_id = int(box_list[5])
                    if cls_id < len(label_names):
                        cat = label_names[cls_id].strip()
                        
                        # [调试功能] 绘制所有识别到的候选框(蓝色)，方便看模型有没有"瞎"
                        if cat == CURRENT_TARGET:
                            bx1 = int(box_list[0]*scale_w); by1 = int(box_list[1]*scale_h)
                            bx2 = int(box_list[2]*scale_w); by2 = int(box_list[3]*scale_h)
                            # 蓝色细框表示：YOLO 看到了这个，但是还没锁定
                            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 0, 0), 1)
                            cv2.putText(frame, f"{cat}", (bx1, by1-5), 0, 0.5, (255,0,0), 1)

                            b_w = box_list[2] - box_list[0]
                            b_h = box_list[3] - box_list[1]
                            area_ratio = (b_w * b_h) / (cfg["width"] * cfg["height"])
                            
                            # --- 差异化过滤逻辑 ---
                            if CURRENT_TARGET == "person":
                                # 人：严格过滤，防止把墙当人
                                aspect_ok = b_h > (b_w * 0.8) # 必须是瘦高的
                                if area_ratio < MAX_LOCK_AREA and aspect_ok:
                                    yolo_boxes.append(box_list)
                            else:
                                # 物体：极度宽容！
                                # 只要不是大得离谱(比如误检了整个地板)，都要
                                if area_ratio < 0.9: 
                                    yolo_boxes.append(box_list)
            
            # 追踪逻辑
            if not is_tracking:
                if len(yolo_boxes) > 0:
                    # 找离中心最近的
                    def dist_center(b):
                        cx = (b[0]+b[2])/2/cfg["width"]; cy = (b[1]+b[3])/2/cfg["height"]
                        return (cx-0.5)**2 + (cy-0.5)**2
                    best_box = min(yolo_boxes, key=dist_center)
                    
                    print(f"🎯 [视觉] 锁定目标: {CURRENT_TARGET}")
                    tracker = cv2.TrackerCSRT_create()
                    x1 = int(best_box[0]*scale_w); y1 = int(best_box[1]*scale_h)
                    wb = int((best_box[2]-best_box[0])*scale_w); hb = int((best_box[3]-best_box[1])*scale_h)
                    tracker.init(frame, (x1, y1, wb, hb))
                    is_tracking = True
                else:
                    # 没找到目标时，显示正在寻找
                    cv2.putText(frame, f"SEARCHING: {CURRENT_TARGET}...", (20, 80), 0, 0.7, (0, 255, 255), 2)
            else:
                frames_since_detect += 1
                ok, bbox = tracker.update(frame)
                if ok:
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3]))
                    # 绿色粗框：表示正在追踪
                    cv2.rectangle(frame, p1, p2, (0, 255, 0), 3)
                    cv2.putText(frame, f"LOCKED: {CURRENT_TARGET}", (p1[0], p1[1]-10), 0, 0.7, (0,255,0), 2)
                    
                    box_for_pid = [bbox[0]/scale_w, bbox[1]/scale_h, (bbox[0]+bbox[2])/scale_w, (bbox[1]+bbox[3])/scale_h]
                    move_cmd, steer_cmd = get_pid_command(box_for_pid, cfg["height"], cfg["width"])
                    send_command(move_cmd, steer_cmd, 0.35 if move_cmd == "FORWARD" else 0.0)
                else:
                    is_tracking = False
                    tracker = None
                    send_command("STOP", 0.0, 0.0)
        else:
            if is_tracking: is_tracking = False; tracker = None
            cv2.putText(frame, "VOICE MODE", (20, 40), 0, 1.0, (0, 0, 255), 2)

        cv2.imshow('SmartCar Vision (Blue=Candidate, Green=Locked)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            PROGRAM_RUNNING = False
            break
            
    cap.release()
    cv2.destroyAllWindows()

# ==========================================
#           模块 2: 语音控制 (v3 增强版)
# ==========================================

SYSTEM_PROMPT = """
你是一个智能小车。用户说中文，你输出JSON。
{ "command": "FORWARD/BACKWARD/STOP", "steer": -1.0到1.0, "throttle": 0.0到1.0, "target": "person/cup/bottle/..." }
"""

def send_command(cmd, steer, throttle):
    try:
        data = {"command": cmd, "steer": steer, "throttle": throttle}
        requests.post(CONTROL_URL, json=data, timeout=0.2)
    except: pass 

def parse_local_logic(text: str):
    """
    极速本地解析
    """
    cmd = {"mode_switch": None, "command": "STOP", "steer": 0.0, "throttle": 0.0, "new_target": None, "unlock": False}
    
    # 0. 优先处理：强制解锁/取消
    if any(w in text for w in ["解锁", "取消", "别追", "松手", "放开"]):
        cmd["unlock"] = True
        return cmd

    # 1. 切换追踪目标
    detected_obj = None
    for cn_key, en_val in CN_COCO_MAP.items():
        if cn_key in text:
            detected_obj = en_val
            break 
            
    if ("追" in text or "跟" in text or "找" in text) and detected_obj:
        cmd["mode_switch"] = "TRACK"
        cmd["new_target"] = detected_obj
        return cmd 

    if "追踪" in text or "自动" in text or "跟我走" in text:
        cmd["mode_switch"] = "TRACK"
        return cmd

    # 2. 回手动模式
    if any(w in text for w in ["手动", "听我", "停止追踪"]):
        cmd["mode_switch"] = "VOICE"
        return cmd

    # 3. 运动指令
    is_motion = any(w in text for w in ["前", "后", "左", "右", "停", "快", "慢", "退", "走"])
    if is_motion:
        cmd["command"] = "FORWARD"
        cmd["throttle"] = 0.35
        if "停" in text or "刹" in text or "别动" in text:
            cmd["command"] = "STOP"; cmd["throttle"] = 0.0; return cmd
        if "后" in text or "退" in text: cmd["command"] = "BACKWARD"
        if "左" in text: cmd["steer"] = -1.0
        elif "右" in text: cmd["steer"] = 1.0
        if "快" in text: cmd["throttle"] = 0.6
        if "慢" in text: cmd["throttle"] = 0.2
        if "一点" in text or "微" in text: 
            cmd["throttle"] = 0.2; 
            if cmd["steer"] != 0: cmd["steer"] *= 0.3 
        return cmd

    return None

def get_llm_command(text: str):
    payload = {
        "model": LLM_MODEL_NAME,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}],
        "stream": False, "temperature": 0.1, "response_format": {"type": "json_object"}
    }
    try:
        print(f"🤖 DeepSeek 思考: '{text}' ...")
        res = session.post(API_URL, json=payload, timeout=3)
        r = res.json()['choices'][0]['message']['content'].strip()
        if "```" in r: r = r.replace("```json", "").replace("```", "")
        return json.loads(r)
    except Exception as e:
        print(f"❌ LLM Error: {e}")
        return None

def audio_loop():
    global CURRENT_MODE, PROGRAM_RUNNING, CURRENT_TARGET, FORCE_UNLOCK_SIGNAL
    
    print("🎙️ 初始化语音...")
    try:
        asr_model = AutoModel(model=ASR_MODEL_PATH, device="cuda" if torch.cuda.is_available() else "cpu", disable_update=True, log_level="ERROR")
        vad_config = VadModelConfig(SileroVadModelConfig(model=VAD_PATH, min_silence_duration=0.5, threshold=0.5), sample_rate=16000)
        vad = VoiceActivityDetector(vad_config, buffer_size_in_seconds=100)
        print("✅ 语音就绪")
    except Exception as e:
        print(f"❌ 语音加载失败: {e}"); return

    sample_rate = 16000
    samples_per_read = int(0.1 * sample_rate)
    
    with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
        while PROGRAM_RUNNING:
            samples, _ = s.read(samples_per_read)
            samples = samples.reshape(-1)
            vad.accept_waveform(samples)
            
            if not vad.empty():
                audio_segment = np.array(vad.front.samples)
                vad.pop()
                if len(audio_segment) > 0:
                    try:
                        res = asr_model.generate(input=[audio_segment], cache={}, language="zh", use_itn=True, batch_size_s=60)
                        text = res[0].get("text", "") if res else ""
                        text = re.sub(r'<\|.*?\|>', '', text).strip()
                        
                        if len(text) > 0:
                            print(f"\n👂: {text}")
                            cmd_data = parse_local_logic(text)
                            if cmd_data is None: cmd_data = get_llm_command(text)
                            if not cmd_data: continue

                            # 1. 强制解锁
                            if cmd_data.get("unlock"):
                                FORCE_UNLOCK_SIGNAL = True
                                print("🔓 正在接触锁定...")
                                continue

                            # 2. 切换目标
                            if cmd_data.get("new_target"):
                                CURRENT_TARGET = cmd_data["new_target"]
                                FORCE_UNLOCK_SIGNAL = True # 切换目标时也先解锁旧的
                                print(f"🎯 目标切换为: {CURRENT_TARGET}")
                            
                            # 3. 切换模式
                            if cmd_data.get("mode_switch"):
                                new_mode = cmd_data["mode_switch"]
                                if new_mode != CURRENT_MODE:
                                    CURRENT_MODE = new_mode
                                    print(f"🔀 模式: {CURRENT_MODE}")
                                    send_command("STOP", 0.0, 0.0)
                                    continue 
                            
                            if CURRENT_MODE == "VOICE":
                                send_command(cmd_data.get("command", "STOP"), cmd_data.get("steer", 0.0), cmd_data.get("throttle", 0.0))
                                
                    except Exception as e:
                        print(f"Error: {e}")

if __name__ == "__main__":
    t_video = Thread(target=video_loop, daemon=True)
    t_video.start()
    try:
        audio_loop()
    except KeyboardInterrupt: pass
    finally:
        PROGRAM_RUNNING = False
        send_command("STOP", 0.0, 0.0)