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
from contextlib import contextmanager

# --- 引入语音依赖 ---
from funasr import AutoModel
from sherpa_onnx import VadModelConfig, SileroVadModelConfig, VoiceActivityDetector

# --- [新增] 引入 ModelScope 用于本地语义嵌入 ---
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

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
# [新增] 本地语义模型 (阿里巴巴达摩院出品，轻量高效)
EMBEDDING_MODEL_PATH = "damo/nlp_corom_sentence-embedding_chinese-base"

CN_COCO_MAP = {
    "人": "person", "我": "person", "自己": "person",
    "瓶": "bottle", "水": "bottle", "杯": "cup",
    "手机": "cell phone", "电话": "cell phone",
    "书": "book", "猫": "cat", "狗": "dog",
    "键盘": "keyboard", "鼠标": "mouse",
    "剪刀": "scissors", "遥控": "remote"
}

CURRENT_MODE = "VOICE"
CURRENT_TARGET = "person"
PROGRAM_RUNNING = True
FORCE_UNLOCK_SIGNAL = False
LAST_CMD_STATE = {"command": "STOP", "steer": 0.0, "throttle": 0.0}

video_lock = Lock()
latest_frame = None
latest_ret = False

session = requests.Session()
session.headers.update({"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"})

# ==========================================
#           核心工具：静音器
# ==========================================
@contextmanager
def no_print():
    """屏蔽终端输出"""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout; old_stderr = sys.stderr
        try:
            sys.stdout = devnull; sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout; sys.stderr = old_stderr

# ==========================================
#       [核心升级] 本地 AI 语义路由器
# ==========================================
class SemanticRouter:
    def __init__(self):
        print("🧠 正在加载本地语义理解模型 (首次需下载)...")
        try:
            # 加载 embedding 模型
            self.pipeline = pipeline(Tasks.sentence_embedding, model=EMBEDDING_MODEL_PATH)
            
            # 定义标准指令库 (锚点)
            # 你的话会和这些锚点比对，而不只是查关键字
            self.command_anchors = {
                "UNLOCK": ["解除锁定", "停止追踪", "别追了", "不要跟了", "取消", "松手"],
                "TRACK_MODE": ["开启追踪", "切换到追踪模式", "跟我走", "自动跟随", "开始找人"],
                "VOICE_MODE": ["切换到语音模式", "手动控制", "听我指挥", "改为语言控制", "回来"],
                "FORWARD": ["前进", "往前走", "向前", "走起来", "Go"],
                "BACKWARD": ["后退", "倒车", "往后退", "退回来"],
                "LEFT": ["左转", "往左拐", "向左"],
                "RIGHT": ["右转", "往右拐", "向右"],
                "STOP": ["停止", "停下", "别动", "刹车", "立定"],
                "SPEED_UP": ["太慢了", "加速", "快点跑", "提速"],
                "SLOW_DOWN": ["太快了", "减速", "慢一点", "慢点跑"]
            }
            
            # 预计算锚点的向量
            self.anchor_embeddings = {}
            for intent, texts in self.command_anchors.items():
                # 计算一组锚点的平均向量
                embeddings = self.get_embeddings(texts)
                # 简单平均作为该意图的中心向量
                self.anchor_embeddings[intent] = np.mean(embeddings, axis=0)
                
            print("✅ 语义大脑构建完成！")
            self.ready = True
        except Exception as e:
            print(f"❌ 语义模型加载失败: {e}")
            self.ready = False

    def get_embeddings(self, texts):
        if isinstance(texts, str): texts = [texts]
        # modelscope 的 pipeline 输入格式
        inputs = {'source_sentence': texts}
        result = self.pipeline(input=inputs)
        return np.array([x['embedding'] for x in result['output']])

    def predict(self, text, threshold=0.60):
        """
        输入文本，返回 (Intent, Score)
        如果最高分低于 threshold，说明是复杂指令，返回 None
        """
        if not self.ready: return None, 0.0
        
        # 1. 计算输入文本的向量
        input_emb = self.get_embeddings(text)[0]
        
        # 2. 计算与所有意图的余弦相似度
        best_intent = None
        best_score = -1.0
        
        for intent, anchor_emb in self.anchor_embeddings.items():
            # Cosine Similarity
            score = np.dot(input_emb, anchor_emb) / (np.linalg.norm(input_emb) * np.linalg.norm(anchor_emb))
            if score > best_score:
                best_score = score
                best_intent = intent
        
        if best_score >= threshold:
            return best_intent, best_score
        else:
            return None, best_score

# 初始化路由器
router = SemanticRouter()

# ==========================================
#           模块 1: 视觉追踪 (v9版逻辑)
# ==========================================
# (为了节省篇幅，这里保持 v9 的 YOLO 代码逻辑不变，功能完全一致)
Kp = 0.35; Ki = 0.0; Kd = 1.0; CENTER_DEAD_ZONE = 0.12; STOP_AREA_THRESHOLD = 0.35; MAX_LOCK_AREA = 0.7; CONF_THRESHOLD = 0.35; ERROR_BUFFER_SIZE = 3
integral = 0.0; previous_error = 0.0; last_pid_time = time.time(); error_buffer = []

def load_yolo_model():
    print("📷 正在加载本地 YOLOv5s 模型 (离线模式)...")
    try:
        model = torch.hub.load('./yolov5', 'custom', path='yolov5s.pt', source='local')
        model.conf = CONF_THRESHOLD; model.iou = 0.45
        if torch.cuda.is_available(): model.cuda()
        print("✅ YOLOv5s 加载成功！")
        return model
    except Exception as e:
        print(f"❌ 视觉模型加载失败: {e}"); return None

def get_pid_command(box, h, w):
    global integral, previous_error, last_pid_time, error_buffer
    x_center = (box[0] + box[2]) / 2 / w
    box_area = (box[2] - box[0]) * (box[3] - box[1]) / (h * w)
    if box_area > STOP_AREA_THRESHOLD: integral=0.0; previous_error=0.0; error_buffer=[]; return "STOP", 0.0
    curr_time = time.time(); dt = curr_time - last_pid_time; 
    if dt == 0: dt = 1e-5
    error_buffer.append(x_center - 0.5)
    if len(error_buffer) > ERROR_BUFFER_SIZE: error_buffer.pop(0)
    smooth_error = sum(error_buffer) / len(error_buffer)
    if abs(smooth_error) < CENTER_DEAD_ZONE: smooth_error = 0.0; previous_error = 0.0
    integral += smooth_error * dt; integral = max(-1.0, min(1.0, integral))
    derivative = (smooth_error - previous_error) / dt
    active_Kp = 0.55 if CURRENT_TARGET != "person" else 0.35
    steer = (active_Kp * smooth_error) + (Ki * integral) + (Kd * derivative)
    previous_error = smooth_error; last_pid_time = curr_time
    return "FORWARD", max(-1.0, min(1.0, steer))

def video_loop():
    global latest_frame, latest_ret, CURRENT_MODE, PROGRAM_RUNNING, CURRENT_TARGET, FORCE_UNLOCK_SIGNAL
    yolo = load_yolo_model()
    if not yolo: return
    cap = cv2.VideoCapture(STREAM_URL)
    def read_stream():
        global latest_frame, latest_ret
        while PROGRAM_RUNNING:
            try:
                ret, frame = cap.read()
                with video_lock: latest_frame = frame; latest_ret = ret
                if not ret: time.sleep(0.5)
                else: time.sleep(0.01)
            except: pass
    Thread(target=read_stream, daemon=True).start()
    tracker = None; is_tracking = False; frames_since = 0
    print("✅ 视觉线程启动 (v10 AI版)")
    while PROGRAM_RUNNING:
        try:
            with video_lock: frame = latest_frame.copy() if latest_frame is not None else None
            if frame is None: time.sleep(0.1); continue
            h, w, _ = frame.shape
            if FORCE_UNLOCK_SIGNAL: is_tracking = False; tracker = None; FORCE_UNLOCK_SIGNAL = False; send_command("STOP", 0, 0); print("🔓 视觉已重置")
            if CURRENT_MODE == "TRACK":
                need_detect = (not is_tracking) or (frames_since > 10)
                cands = []
                if need_detect:
                    frames_since = 0
                    try:
                        results = yolo(frame)
                        for det in results.xyxy[0].cpu().numpy():
                            x1, y1, x2, y2, conf, cls = det
                            if int(cls) < len(results.names):
                                name = results.names[int(cls)]
                                if name == CURRENT_TARGET:
                                    cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (255,0,0), 1)
                                    cv2.putText(frame, f"{name} {conf:.2f}", (int(x1), int(y1)-5), 0, 0.5, (255,0,0), 1)
                                    b_w = x2-x1; b_h = y2-y1; ratio = (b_w*b_h)/(w*h)
                                    if CURRENT_TARGET == "person":
                                        if ratio < MAX_LOCK_AREA and b_h > (b_w * 0.7): cands.append([x1,y1,x2,y2])
                                    else:
                                        if ratio < 0.9: cands.append([x1,y1,x2,y2])
                    except: pass
                if not is_tracking:
                    if cands:
                        best = min(cands, key=lambda b: ((b[0]+b[2])/2/w-0.5)**2 + ((b[1]+b[3])/2/h-0.5)**2)
                        tracker = cv2.TrackerCSRT_create()
                        tracker.init(frame, (int(best[0]), int(best[1]), int(best[2]-best[0]), int(best[3]-best[1])))
                        is_tracking = True; print(f"🎯 视觉锁定: {CURRENT_TARGET}")
                    else: cv2.putText(frame, f"SEARCHING: {CURRENT_TARGET}...", (20, 80), 0, 0.7, (0,255,255), 2)
                else:
                    frames_since += 1
                    ok, bbox = tracker.update(frame)
                    if ok:
                        p1 = (int(bbox[0]), int(bbox[1])); p2 = (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3]))
                        cv2.rectangle(frame, p1, p2, (0,255,0), 3)
                        cv2.putText(frame, "LOCKED", (p1[0], p1[1]-10), 0, 0.7, (0,255,0), 2)
                        cmd, steer = get_pid_command([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]], h, w)
                        send_command(cmd, steer, 0.35 if cmd=="FORWARD" else 0.0)
                    else: is_tracking = False; tracker = None; send_command("STOP", 0, 0)
            else:
                if is_tracking: is_tracking=False; tracker=None
                cv2.putText(frame, "VOICE MODE", (20, 40), 0, 1.0, (0,0,255), 2)
            cv2.imshow('SmartCar Vision v10', frame)
            if cv2.waitKey(1) == ord('q'): PROGRAM_RUNNING=False; break
        except Exception as e: time.sleep(1)
    cap.release(); cv2.destroyAllWindows()

# ==========================================
#           模块 2: 语音控制 (语义路由 + LLM)
# ==========================================

SYSTEM_PROMPT = """
你是一个智能小车的控制大脑。请将用户的口语指令转换为 JSON 控制信号。
{ "command": "FORWARD/BACKWARD/STOP", "steer": -1.0到1.0, "throttle": 0.0到1.0, "target": "person/cup/bottle/..." }
"""

def send_command(cmd, steer, throttle):
    global LAST_CMD_STATE
    LAST_CMD_STATE = {"command": cmd, "steer": steer, "throttle": throttle}
    try: requests.post(CONTROL_URL, json=LAST_CMD_STATE, timeout=0.2)
    except: pass 

def handle_ai_command(text):
    """
    语义处理核心逻辑
    """
    cmd_data = {"mode_switch": None, "command": "STOP", "steer": 0.0, "throttle": 0.0, "new_target": None, "unlock": False}
    
    # --- 1. 使用本地 AI 模型判断意图 ---
    intent, score = router.predict(text)
    
    # 打印匹配结果供调试
    if intent:
        print(f"🧠 本地AI判断: [{intent}] (置信度: {score:.2f})")
    
    # 如果置信度高，直接执行本地预设逻辑
    if intent and score > 0.65:
        # 模式与解锁
        if intent == "UNLOCK": cmd_data["unlock"] = True; return cmd_data
        if intent == "TRACK_MODE": cmd_data["mode_switch"] = "TRACK"; return cmd_data
        if intent == "VOICE_MODE": cmd_data["mode_switch"] = "VOICE"; return cmd_data
        
        # 运动控制
        current_steer = 0.0
        current_throttle = 0.35
        current_cmd = "FORWARD"
        
        # 继承旧状态
        if intent in ["SPEED_UP", "SLOW_DOWN"]:
            current_cmd = LAST_CMD_STATE.get("command", "FORWARD")
            if current_cmd == "STOP": current_cmd = "FORWARD"
            current_steer = LAST_CMD_STATE.get("steer", 0.0)
            
        if intent == "FORWARD": current_cmd = "FORWARD"
        elif intent == "BACKWARD": current_cmd = "BACKWARD"
        elif intent == "STOP": current_cmd = "STOP"
        
        if intent == "LEFT": current_steer = -1.0; current_cmd = "FORWARD"
        elif intent == "RIGHT": current_steer = 1.0; current_cmd = "FORWARD"
        
        if intent == "SPEED_UP": current_throttle = 0.6
        elif intent == "SLOW_DOWN": current_throttle = 0.2
            
        cmd_data["command"] = current_cmd
        cmd_data["steer"] = current_steer
        cmd_data["throttle"] = current_throttle
        return cmd_data
    
    # --- 2. 目标检测切换 (仍保留简单的关键词，因为物体名字太多) ---
    tgt = next((v for k,v in CN_COCO_MAP.items() if k in text), None)
    if ("追" in text or "跟" in text or "找" in text) and tgt:
        cmd_data["mode_switch"]="TRACK"
        cmd_data["new_target"]=tgt
        return cmd_data

    # --- 3. 如果本地 AI 觉得不像任何标准指令，交给 LLM ---
    print("🤔 本地模型没听懂，请求 LLM 支援...")
    return None # 返回 None 触发外部 LLM 调用

def audio_loop():
    global CURRENT_MODE, PROGRAM_RUNNING, CURRENT_TARGET, FORCE_UNLOCK_SIGNAL
    print("🎙️ 初始化语音 (v10 AI语义增强版)...")
    try:
        asr = AutoModel(model=ASR_MODEL_PATH, disable_update=True, log_level="ERROR")
        vad = VoiceActivityDetector(VadModelConfig(SileroVadModelConfig(model=VAD_PATH, min_silence_duration=0.5, threshold=0.5), sample_rate=16000), buffer_size_in_seconds=100)
        print("✅ 语音就绪")
    except Exception as e: print(f"❌ 语音挂了: {e}"); return

    sr = 16000; batch = int(0.1*sr)
    with sd.InputStream(channels=1, dtype="float32", samplerate=sr) as s:
        while PROGRAM_RUNNING:
            d, _ = s.read(batch); d = d.reshape(-1)
            vad.accept_waveform(d)
            if not vad.empty():
                raw = np.array(vad.front.samples); vad.pop()
                if len(raw) > 0:
                    try:
                        with no_print(): 
                            res = asr.generate(input=[raw], cache={}, language="zh", use_itn=True, batch_size_s=60)
                        
                        text = res[0].get("text", "") if res else ""; text = re.sub(r'<\|.*?\|>', '', text).strip()
                        if text:
                            print(f"\n👂: {text}")
                            
                            # 调用 AI 路由逻辑
                            data = handle_ai_command(text)
                            
                            # 如果本地 AI 搞不定，呼叫 DeepSeek
                            if not data and len(text) > 1: 
                                payload = {"model": LLM_MODEL_NAME, "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}], "stream": False, "response_format": {"type": "json_object"}}
                                try: 
                                    r = session.post(API_URL, json=payload, timeout=3).json()['choices'][0]['message']['content']
                                    data = json.loads(r.replace("```json","").replace("```",""))
                                    print(f"🤖 LLM 响应: {data}")
                                except: pass

                            if not data: continue
                            
                            if data.get("unlock"): FORCE_UNLOCK_SIGNAL=True; print("🔓 指令: 解除追踪"); continue
                            if data.get("new_target"): CURRENT_TARGET=data["new_target"]; FORCE_UNLOCK_SIGNAL=True; print(f"🎯 新目标: {CURRENT_TARGET}")
                            if data.get("mode_switch"): 
                                if data["mode_switch"] != CURRENT_MODE: CURRENT_MODE=data["mode_switch"]; print(f"🔀 模式: {CURRENT_MODE}"); send_command("STOP",0,0); continue
                            
                            if CURRENT_MODE=="VOICE": 
                                send_command(data.get("command","STOP"), data.get("steer",0), data.get("throttle",0))
                    except Exception as e: 
                        if "WinError 6" not in str(e): print(f"语音错误: {e}")

if __name__ == "__main__":
    t = Thread(target=video_loop, daemon=True); t.start()
    try: audio_loop()
    except KeyboardInterrupt: pass
    finally: PROGRAM_RUNNING=False; send_command("STOP",0,0)