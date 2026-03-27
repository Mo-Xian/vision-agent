"""主窗口：训练工坊（行为克隆）+ 自对弈（RL）+ LLM 配置。"""

import os
import json
import threading
from pathlib import Path
from PySide6.QtCore import Qt, Slot, Signal, QSettings, QTimer
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QTextEdit,
    QMessageBox, QStackedWidget,
)

from ..decision import PROVIDER_PRESETS, create_provider
from .workshop_panel import WorkshopPanel
from .selfplay_panel import SelfPlayPanel
from .llm_panel import LLMPanel
from .styles import MAIN_STYLESHEET, COLORS

# 模式切换按钮样式
_MODE_BTN_STYLE = f"""
QPushButton#modeBtnActive {{
    background-color: {COLORS['accent']};
    color: white;
    font-size: 14px;
    font-weight: bold;
    border-radius: 8px;
    padding: 10px 0px;
    border: none;
}}
QPushButton#modeBtnInactive {{
    background-color: {COLORS['bg_card']};
    color: {COLORS['text_secondary']};
    font-size: 14px;
    font-weight: bold;
    border-radius: 8px;
    padding: 10px 0px;
    border: 1px solid {COLORS['border']};
}}
QPushButton#modeBtnInactive:hover {{
    color: {COLORS['text']};
    border-color: {COLORS['border_hover']};
    background-color: {COLORS['bg_input']};
}}
"""


class MainWindow(QMainWindow):
    _learn_log = Signal(str)
    _learn_progress = Signal(str, float)
    _learn_done = Signal(dict)
    _recording_done = Signal(str, dict)  # rec_dir, stats
    _selfplay_log = Signal(str)
    _selfplay_stats = Signal(dict)
    _selfplay_frame = Signal(object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vision Agent")

        from PySide6.QtGui import QGuiApplication
        screen = QGuiApplication.primaryScreen()
        if screen:
            avail = screen.availableGeometry()
            w = min(1200, int(avail.width() * 0.75))
            h = min(820, int(avail.height() * 0.85))
            self.setMinimumSize(min(900, int(avail.width() * 0.6)),
                                min(600, int(avail.height() * 0.6)))
            self.resize(w, h)
            self.move(avail.x() + (avail.width() - w) // 2,
                      avail.y() + (avail.height() - h) // 2)
        else:
            self.setMinimumSize(700, 500)
            self.resize(1000, 720)

        self.setStyleSheet(MAIN_STYLESHEET + _MODE_BTN_STYLE)
        self.setAcceptDrops(True)

        self._settings = QSettings("VisionAgent", "VisionAgent")
        self._current_mode = "workshop"
        self._learner = None
        self._recorder = None
        self._selfplay_loop = None
        self._selfplay_capture_timer = None
        self._dqn_engine = None
        self._agent_running = False
        self._last_model_dir = ""

        self._init_ui()
        self._connect_signals()
        self._load_settings()
        self.workshop_panel.init_scene_manager()

    # ================================================================
    #  UI 构建
    # ================================================================

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        # 模式切换
        mode_row = QHBoxLayout()
        mode_row.setSpacing(4)
        self.mode_workshop_btn = QPushButton("训练工坊")
        self.mode_selfplay_btn = QPushButton("Agent 部署")
        self.mode_llm_btn = QPushButton("LLM 设置")
        for btn, mode in [
            (self.mode_workshop_btn, "workshop"),
            (self.mode_selfplay_btn, "selfplay"),
            (self.mode_llm_btn, "llm"),
        ]:
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked=False, m=mode: self._switch_mode(m))
            mode_row.addWidget(btn)
        mode_row.addStretch()
        root_layout.addLayout(mode_row)

        # 模式面板栈
        self.mode_stack = QStackedWidget()
        self.workshop_panel = WorkshopPanel()
        self.selfplay_panel = SelfPlayPanel()
        self.llm_panel = LLMPanel()
        self.mode_stack.addWidget(self.workshop_panel)  # 0
        self.mode_stack.addWidget(self.selfplay_panel)  # 1
        self.mode_stack.addWidget(self.llm_panel)       # 2
        root_layout.addWidget(self.mode_stack, 1)

        # 底部日志
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        self.log_text.setPlaceholderText("系统日志...")
        root_layout.addWidget(self.log_text)

        self._switch_mode("workshop")

    # ================================================================
    #  信号连接
    # ================================================================

    def _connect_signals(self):
        wp = self.workshop_panel
        sp = self.selfplay_panel
        lp = self.llm_panel

        # 学习管线信号
        self._learn_log.connect(self._on_learn_log)
        self._learn_progress.connect(self._on_learn_progress)
        self._learn_done.connect(self._on_learn_done)

        # LLM 面板
        lp.llm_provider_combo.addItems(list(PROVIDER_PRESETS.keys()))
        lp.llm_provider_combo.currentTextChanged.connect(self._on_provider_changed)
        lp.llm_test_btn.clicked.connect(self._test_llm_connection)
        lp.llm_save_btn.clicked.connect(self._save_llm_settings)

        # 训练工坊
        wp.learn_from_recording_requested.connect(self._start_learning_from_recordings)
        wp.stop_requested.connect(self._stop_learning)
        wp.recording_started.connect(self._start_recording)
        wp.recording_stopped.connect(self._stop_recording)
        wp.view_models_btn.clicked.connect(self._view_models)

        # Agent 部署面板
        sp.agent_start_requested.connect(self._start_agent)
        sp.agent_stop_requested.connect(self._stop_agent)
        self._selfplay_log.connect(sp._append_log)
        self._selfplay_log.connect(self._on_learn_log)  # 也显示到工坊日志
        self._selfplay_frame.connect(sp._update_frame)

        # 录制完成信号
        self._recording_done.connect(self._on_recording_done)

        self._on_provider_changed(lp.llm_provider_combo.currentText())

    # ================================================================
    #  模式切换
    # ================================================================

    def _switch_mode(self, mode: str):
        self._current_mode = mode
        index = {"workshop": 0, "selfplay": 1, "llm": 2}.get(mode, 0)
        self.mode_stack.setCurrentIndex(index)

        for btn, m in [
            (self.mode_workshop_btn, "workshop"),
            (self.mode_selfplay_btn, "selfplay"),
            (self.mode_llm_btn, "llm"),
        ]:
            btn.setObjectName("modeBtnActive" if m == mode else "modeBtnInactive")
            btn.setStyle(btn.style())

    @Slot()
    def _stop_learning(self):
        if self._learner:
            self._learner.stop()
        self.workshop_panel.stop_learn_btn.setEnabled(False)

    @Slot(str)
    def _on_learn_log(self, msg: str):
        self.workshop_panel.log_text.append(msg)
        sb = self.workshop_panel.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    @Slot(str, float)
    def _on_learn_progress(self, phase: str, pct: float):
        wp = self.workshop_panel
        phase_labels = {
            "discover": "动作发现",
            "train": "训练模型",
            "expand": "伪标签扩展",
            "rl": "强化学习", "done": "完成",
        }
        label = phase_labels.get(phase, phase)
        wp.progress_bar.setValue(int(pct * 100))
        wp.progress_bar.setFormat(f"%p% - {label}")
        wp.phase_label.setText(f"阶段: {label} ({pct*100:.0f}%)")

    @Slot(dict)
    def _on_learn_done(self, result: dict):
        wp = self.workshop_panel
        wp.set_learning_state(False)
        self._learner = None

        scene = wp.current_scene

        if result.get("model_dir"):
            wp.progress_bar.setValue(100)
            wp.progress_bar.setFormat("100% - 学习完成")
            wp.model_status.setText(f"模型已生成: {result['model_dir']}")
            self._last_model_dir = result["model_dir"]

            insight = result.get("insight")
            if insight:
                wp.update_insight(insight)

            if scene:
                val_acc = result.get("metrics", {}).get("best_val_acc", 0)
                scene.total_annotated += result.get("annotated_count", 0)
                scene.update_best_model(
                    result["model_dir"], val_acc,
                    result.get("profile_path", ""),
                )
                wp._update_scene_ui(scene)
                wp._refresh_scene_list()

            self._log(f"[工坊] 学习完成 → {result['model_dir']}")

            # 显示教练建议
            coach = result.get("coach_advice", {})
            advice_text = ""
            if coach.get("suggestions"):
                advice_text = "\n\n教练建议:\n" + "\n".join(
                    f"  - {s}" for s in coach["suggestions"][:3]
                )

            QMessageBox.information(
                self, "学习完成",
                f"模型: {result['model_dir']}\n"
                f"Profile: {result.get('profile_path', 'N/A')}"
                f"{advice_text}\n\n"
                f"使用 eval_model.py 评估模型效果。"
            )
        else:
            wp.progress_bar.setFormat("失败或中止")
            if scene:
                scene.status = "idle"
                scene.save()
            self._log("[工坊] 学习失败或中止")

    # ================================================================
    #  录制
    # ================================================================

    @Slot()
    def _start_recording(self):
        if self._recorder and self._recorder.is_recording:
            return

        wp = self.workshop_panel
        scene = wp.current_scene

        # 确定输出目录
        import time as _time
        if scene:
            rec_dir = str(Path(scene.scene_dir) / "recordings" / _time.strftime("%Y%m%d_%H%M%S"))
        else:
            rec_dir = f"recordings/{_time.strftime('%Y%m%d_%H%M%S')}"

        from ..data.game_recorder import GameRecorder
        window_title = wp.get_window_title()
        self._recorder = GameRecorder(
            output_dir=rec_dir,
            fps=10,
            window_title=window_title,
            record_mouse=True,
            on_log=lambda msg: self._learn_log.emit(msg),
        )
        self._recorder.start()
        wp.set_recording_state(True)
        window_info = f" (窗口: {window_title})" if window_title else " (全屏)"
        self._log(f"[录制] 开始录制{window_info}，F9 暂停/恢复...")

    @Slot()
    def _stop_recording(self):
        if not self._recorder or not self._recorder.is_recording:
            return

        wp = self.workshop_panel
        wp.set_recording_state(False)
        wp.record_status.setText("保存中...")

        recorder = self._recorder

        def _save():
            try:
                stats = recorder.stop()
                self._recording_done.emit(
                    stats.output_dir,
                    {
                        "total_frames": stats.total_frames,
                        "total_events": stats.total_events,
                        "duration_sec": stats.duration_sec,
                        "action_dist": stats.action_dist,
                    },
                )
            except Exception as e:
                self._learn_log.emit(f"[录制] 保存失败: {e}")

        threading.Thread(target=_save, daemon=True).start()

    @Slot(str, dict)
    def _on_recording_done(self, rec_dir: str, stats: dict):
        wp = self.workshop_panel
        wp.set_recording_state(False)
        wp.record_status.setText("")
        wp.add_recording(rec_dir, stats)
        self._recorder = None

        self._log(
            f"[录制] 完成: {stats.get('total_frames', 0)} 帧, "
            f"{stats.get('duration_sec', 0):.0f}s"
        )
        QMessageBox.information(
            self, "录制完成",
            f"帧数: {stats.get('total_frames', 0)}\n"
            f"时长: {stats.get('duration_sec', 0):.0f}s\n"
            f"事件: {stats.get('total_events', 0)}\n\n"
            f"已添加到录制列表，可继续录制更多或开始学习。"
        )

    @Slot()
    def _start_learning_from_recordings(self):
        wp = self.workshop_panel
        recording_dirs = wp.get_recording_dirs()
        if not recording_dirs:
            QMessageBox.warning(self, "提示", "请先录制操作或导入录制数据")
            return

        api_key = self._get_llm_api_key()
        provider_name = self.llm_panel.llm_provider_combo.currentText()

        wp.set_learning_state(True)
        wp.log_text.clear()
        wp.progress_bar.setValue(0)
        wp.train_chart.clear()
        wp.train_chart.setVisible(True)

        from ..workshop.unified_pipeline import UnifiedPipeline

        output_dir = "runs/workshop"
        if wp.current_scene:
            output_dir = str(Path(wp.current_scene.scene_dir) / "sessions")
            wp.current_scene.session_count += 1
            wp.current_scene.status = "training"
            wp.current_scene.save()

        # 获取自对弈配置（如果有设备连接）
        device_serial = self.selfplay_panel.get_device_serial()
        sp_config = wp.get_selfplay_config()

        self._learner = UnifiedPipeline(
            llm_provider_name=provider_name,
            llm_api_key=api_key,
            llm_model=self.llm_panel.llm_model_combo.currentText(),
            llm_base_url=self.llm_panel.llm_base_url.text().strip(),
            output_dir=output_dir,
            on_log=lambda msg: self._learn_log.emit(msg),
            on_progress=lambda phase, pct, detail="": self._learn_progress.emit(phase, pct),
            on_phase_change=lambda phase: self._learn_log.emit(f"[阶段] → {phase}"),
        )

        kwargs = {
            "recording_dirs": recording_dirs,
            "description": wp.description_input.text().strip(),
            "epochs": wp.epochs_spin.value(),
            "knowledge": wp.knowledge_input.toPlainText().strip(),
            "device_serial": device_serial,
            "selfplay_preset": sp_config.get("preset", ""),
            "selfplay_episodes": sp_config.get("max_episodes", 0),
        }

        def _run():
            try:
                result = self._learner.run(**kwargs)
                self._learn_done.emit(result.to_dict())
            except Exception as e:
                self._learn_log.emit(f"[错误] {e}")
                self._learn_done.emit({})

        threading.Thread(target=_run, daemon=True).start()
        self._log("[工坊] 行为克隆学习开始...")

    @Slot()
    def _view_models(self):
        from ..workshop.model_registry import ModelRegistry
        registry = ModelRegistry()
        count = registry.scan()
        models = registry.list_models()
        if not models:
            QMessageBox.information(self, "模型列表", "未找到训练产出的模型")
            return
        lines = [f"共找到 {count} 个模型:\n"]
        for m in models[:20]:
            lines.append(
                f"  {m.name}  |  val_acc={m.val_acc:.3f}  |  "
                f"type={m.model_type}  |  samples={m.train_samples}  |  {m.trained_at}"
            )
        QMessageBox.information(self, "模型列表", "\n".join(lines))

    # ================================================================
    #  自对弈训练（在训练工坊中）
    # ================================================================

    @Slot()
    def _start_selfplay(self):
        if self._selfplay_loop:
            return

        wp = self.workshop_panel
        sp_config = wp.get_selfplay_config()
        # 设备信息从 Agent 部署面板获取
        device_serial = self.selfplay_panel.get_device_serial()

        if not device_serial:
            QMessageBox.warning(
                self, "提示",
                "请先在「Agent 部署」面板中连接手机设备。"
            )
            return

        def _run():
            try:
                from ..rl.preset import load_selfplay_preset
                from ..rl.self_play import SelfPlayLoop

                preset = load_selfplay_preset(sp_config["preset"])
                dqn = preset.get("dqn_params", {})

                def _on_stats(s):
                    self._selfplay_stats.emit(s)

                self._selfplay_loop = SelfPlayLoop(
                    action_zones=preset["action_zones"],
                    bc_model_dir=sp_config.get("bc_model_dir", "") or preset.get("bc_model_dir", ""),
                    output_dir=preset.get("output_dir", "runs/selfplay/exp1"),
                    device_serial=device_serial,
                    reward_config=preset.get("reward_config"),
                    start_model_path=preset.get("start_model_path", ""),
                    lr=dqn.get("lr", 0.0005),
                    gamma=dqn.get("gamma", 0.99),
                    epsilon_start=dqn.get("epsilon_start", 1.0),
                    epsilon_end=dqn.get("epsilon_end", 0.05),
                    epsilon_decay=dqn.get("epsilon_decay", 0.998),
                    buffer_capacity=dqn.get("buffer_capacity", 50000),
                    batch_size=dqn.get("batch_size", 64),
                    fps=sp_config.get("fps", 5),
                    max_episodes=sp_config.get("max_episodes", 0),
                    on_log=lambda msg: self._selfplay_log.emit(msg),
                    on_stats=_on_stats,
                )

                self._start_frame_capture()
                self._selfplay_loop.start()

                while self._selfplay_loop and self._selfplay_loop.is_running:
                    import time
                    time.sleep(0.5)

            except Exception as e:
                self._selfplay_log.emit(f"[错误] {e}")
            finally:
                self._selfplay_loop = None
                self._stop_frame_capture()
                pass  # selfplay 结束

        threading.Thread(target=_run, daemon=True).start()
        self._log("[自对弈] 启动训练...")

    @Slot()
    def _stop_selfplay(self):
        if self._selfplay_loop:
            self._selfplay_log.emit("[自对弈] 正在停止...")
            loop = self._selfplay_loop
            threading.Thread(target=loop.stop, daemon=True).start()

    def _start_frame_capture(self):
        """启动帧捕获定时器，定期从 scrcpy 窗口截图并发送到面板。"""
        def _capture():
            if not self._selfplay_loop or not self._selfplay_loop.is_running:
                return
            try:
                frame = self._selfplay_loop._env._capture_frame()
                if frame is not None:
                    self._selfplay_frame.emit(frame)
            except Exception:
                pass

        self._selfplay_capture_timer = QTimer()
        self._selfplay_capture_timer.timeout.connect(_capture)
        self._selfplay_capture_timer.start(200)

    def _stop_frame_capture(self):
        if self._selfplay_capture_timer:
            QTimer.singleShot(0, self._selfplay_capture_timer.stop)

    # ================================================================
    #  Agent 部署（用训练模型接管游戏）
    # ================================================================

    @Slot()
    def _start_agent(self):
        if self._agent_running:
            return

        sp = self.selfplay_panel
        model_dir = sp.get_agent_model_dir()
        if not model_dir:
            QMessageBox.warning(self, "提示", "请选择模型目录（DQN 或 BC 训练产出）")
            return

        device_serial = sp.get_device_serial()
        preset_name = sp.get_preset_name()

        sp.set_agent_running_state(True)
        self._agent_running = True

        def _run():
            try:
                from ..rl.preset import load_selfplay_preset
                from ..decision.dqn_engine import DQNEngine

                preset = load_selfplay_preset(preset_name)

                self._dqn_engine = DQNEngine(
                    model_dir=model_dir,
                    touch_zones=preset.get("action_zones", []),
                    device_serial=device_serial,
                    execute_actions=True,
                )
                self._dqn_engine.on_start()

                self._selfplay_log.emit(f"[Agent] 模型加载完成: {model_dir}")
                self._selfplay_log.emit(f"[Agent] 开始接管游戏 (预设: {preset_name})")

                from ..core.vision_encoder import VisionEncoder
                encoder = VisionEncoder()

                # 启动帧捕获显示
                self._start_agent_frame_capture()

                import time
                while self._agent_running:
                    frame = self._capture_scrcpy_frame()
                    if frame is None:
                        time.sleep(0.5)
                        continue

                    self._selfplay_frame.emit(frame)

                    actions = self._dqn_engine.decide(embedding=frame)
                    if actions:
                        action = actions[0]
                        if action.name != "idle":
                            self._selfplay_log.emit(
                                f"[Agent] {action.name} (conf={action.confidence:.2f}) {action.reason}"
                            )

                    time.sleep(0.2)  # ~5 FPS

            except Exception as e:
                self._selfplay_log.emit(f"[Agent 错误] {e}")
            finally:
                if self._dqn_engine:
                    self._dqn_engine.on_stop()
                    self._dqn_engine = None
                self._agent_running = False
                self._stop_frame_capture()
                QTimer.singleShot(0, lambda: sp.set_agent_running_state(False))

        threading.Thread(target=_run, daemon=True).start()
        self._log("[Agent] 启动中...")

    @Slot()
    def _stop_agent(self):
        self._agent_running = False
        self._selfplay_log.emit("[Agent] 正在停止...")

    def _start_agent_frame_capture(self):
        """Agent 模式的帧捕获（复用 scrcpy 窗口）。"""
        pass  # Agent 循环内直接发送帧，无需额外定时器

    def _capture_scrcpy_frame(self):
        """从 scrcpy 窗口截取一帧。"""
        try:
            from ..data.game_recorder import GameRecorder
            region = GameRecorder._find_window("scrcpy")
            if not region:
                return None
            import mss
            import cv2
            import numpy as np
            with mss.mss() as sct:
                img = sct.grab(region)
                frame = np.array(img)
                return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        except Exception:
            return None

    # ================================================================
    #  LLM 设置
    # ================================================================

    @Slot(str)
    def _on_provider_changed(self, provider_name: str):
        lp = self.llm_panel
        preset = PROVIDER_PRESETS.get(provider_name, {})
        lp.llm_model_combo.clear()
        lp.llm_model_combo.addItems(preset.get("models", []))
        lp.llm_base_url.setText(preset.get("base_url", ""))
        env_key = preset.get("api_key_env", "")
        if env_key and os.environ.get(env_key):
            lp.llm_api_key.setPlaceholderText(f"已从 {env_key} 读取")
        else:
            lp.llm_api_key.setPlaceholderText(f"API Key" + (f" 或设置 {env_key}" if env_key else ""))

    @Slot()
    def _test_llm_connection(self):
        lp = self.llm_panel
        provider_name = lp.llm_provider_combo.currentText()
        api_key = self._get_llm_api_key()
        model = lp.llm_model_combo.currentText()
        base_url = lp.llm_base_url.text().strip()
        if not api_key and provider_name != "ollama":
            QMessageBox.warning(self, "提示", "请输入 API Key")
            return
        lp.llm_test_btn.setEnabled(False)
        lp.llm_test_btn.setText("测试中...")
        try:
            provider = create_provider(provider_name, api_key or "ollama", model, base_url)
            ok = provider.test_connection()
            if ok:
                QMessageBox.information(self, "成功", f"{provider_name} 连接成功!\n模型: {model}")
            else:
                QMessageBox.warning(self, "失败", "连接测试失败")
        except ImportError as e:
            pkg = "anthropic" if provider_name == "claude" else "openai"
            QMessageBox.critical(self, "缺少依赖", f"需要安装: pip install {pkg}\n\n{e}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"连接失败:\n{e}")
        finally:
            lp.llm_test_btn.setEnabled(True)
            lp.llm_test_btn.setText("测试连接")

    @Slot()
    def _save_llm_settings(self):
        lp = self.llm_panel
        s = self._settings
        s.setValue("decision/provider", lp.llm_provider_combo.currentText())
        s.setValue("decision/model", lp.llm_model_combo.currentText())
        s.setValue("decision/base_url", lp.llm_base_url.text())
        s.sync()
        lp.save_hint.setText("已保存")
        self._log("[LLM] 配置已保存")
        from PySide6.QtCore import QTimer
        QTimer.singleShot(3000, lambda: lp.save_hint.setText(""))

    def _get_llm_api_key(self) -> str:
        lp = self.llm_panel
        key = lp.llm_api_key.text().strip()
        if key:
            return key
        provider_name = lp.llm_provider_combo.currentText()
        env_key = PROVIDER_PRESETS.get(provider_name, {}).get("api_key_env", "")
        return os.environ.get(env_key, "") if env_key else ""

    # ================================================================
    #  日志
    # ================================================================

    def _log(self, text):
        self.log_text.append(text)
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ================================================================
    #  设置保存 / 加载
    # ================================================================

    def _save_settings(self):
        s = self._settings
        lp = self.llm_panel
        wp = self.workshop_panel
        s.setValue("app/mode", self._current_mode)
        s.setValue("decision/provider", lp.llm_provider_combo.currentText())
        s.setValue("decision/model", lp.llm_model_combo.currentText())
        s.setValue("decision/base_url", lp.llm_base_url.text())
        s.setValue("train/epochs", wp.epochs_spin.value())

    def _load_settings(self):
        s = self._settings
        lp = self.llm_panel
        wp = self.workshop_panel
        if s.value("app/mode"):
            self._switch_mode(s.value("app/mode"))
        if s.value("decision/provider"):
            lp.llm_provider_combo.setCurrentText(s.value("decision/provider"))
        if s.value("decision/model"):
            lp.llm_model_combo.setCurrentText(s.value("decision/model"))
        if s.value("decision/base_url"):
            lp.llm_base_url.setText(s.value("decision/base_url"))
        if s.value("train/epochs") is not None:
            try: wp.epochs_spin.setValue(int(s.value("train/epochs", 100)))
            except (TypeError, ValueError): pass

    def closeEvent(self, event):
        self._save_settings()
        if self._recorder and self._recorder.is_recording:
            self._recorder.stop()
        if self._learner:
            self._learner.stop()
        if self._selfplay_loop:
            self._selfplay_loop.stop()
        self._agent_running = False
        if self._dqn_engine:
            self._dqn_engine.on_stop()
        self._stop_frame_capture()
        event.accept()

    # ================================================================
    #  拖放
    # ================================================================

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        if not path:
            return

        if Path(path).is_dir():
            # 尝试作为录制目录导入
            p = Path(path)
            if (p / "recording.mp4").exists() and (p / "actions.jsonl").exists():
                wp = self.workshop_panel
                meta_path = p / "meta.json"
                stats = None
                if meta_path.exists():
                    try:
                        import json as _json
                        with open(meta_path, "r", encoding="utf-8") as f:
                            stats = _json.load(f)
                    except Exception:
                        pass
                wp.add_recording(path, stats)
                self._log(f"已导入录制: {p.name}")
            else:
                self._log(f"目录不是有效的录制数据: {Path(path).name}")
