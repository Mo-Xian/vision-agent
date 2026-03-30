package com.visionagent.client

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.GestureDescription
import android.graphics.Path
import android.os.Build
import android.view.KeyEvent
import android.view.accessibility.AccessibilityEvent
import org.json.JSONObject

/**
 * 无障碍服务：接收 Agent 控制指令并在本机执行。
 *
 * 支持的指令（与 PC 客户端 RemoteCaptureClient 一致）：
 *   {"cmd":"mouse_click", "x":100, "y":200}       → 点击
 *   {"cmd":"mouse_move",  "x":100, "y":200}       → 移动（忽略，移动端无光标）
 *   {"cmd":"swipe", "x1":100, "y1":200, "x2":300, "y2":400, "duration":300} → 滑动
 *   {"cmd":"key_tap", "key":"back"}                → 按键（back/home/recents）
 *
 * 需要用户在系统设置 → 无障碍中手动启用本服务。
 */
class ControlService : AccessibilityService() {

    companion object {
        var instance: ControlService? = null
            private set

        fun isEnabled(): Boolean = instance != null
    }

    override fun onServiceConnected() {
        super.onServiceConnected()
        instance = this
        android.util.Log.i("ControlService", "无障碍服务已启用")
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        // 不处理事件，只用于执行操控
    }

    override fun onInterrupt() {}

    override fun onDestroy() {
        instance = null
        super.onDestroy()
    }

    /**
     * 执行来自中转服务的控制指令。
     */
    fun executeControl(data: JSONObject) {
        val cmd = data.optString("cmd", "")
        try {
            when (cmd) {
                "mouse_click" -> {
                    val x = data.optDouble("x", 0.0).toFloat()
                    val y = data.optDouble("y", 0.0).toFloat()
                    performTap(x, y)
                }
                "swipe" -> {
                    val x1 = data.optDouble("x1", 0.0).toFloat()
                    val y1 = data.optDouble("y1", 0.0).toFloat()
                    val x2 = data.optDouble("x2", 0.0).toFloat()
                    val y2 = data.optDouble("y2", 0.0).toFloat()
                    val duration = data.optLong("duration", 300)
                    performSwipe(x1, y1, x2, y2, duration)
                }
                "key_tap" -> {
                    val key = data.optString("key", "")
                    performKeyTap(key)
                }
                "mouse_move" -> {
                    // 移动端无光标，忽略
                }
            }
        } catch (e: Exception) {
            android.util.Log.e("ControlService", "执行控制失败: $cmd → ${e.message}")
        }
    }

    private fun performTap(x: Float, y: Float) {
        val path = Path().apply { moveTo(x, y) }
        val gesture = GestureDescription.Builder()
            .addStroke(GestureDescription.StrokeDescription(path, 0, 50))
            .build()
        dispatchGesture(gesture, null, null)
    }

    private fun performSwipe(x1: Float, y1: Float, x2: Float, y2: Float, durationMs: Long) {
        val path = Path().apply {
            moveTo(x1, y1)
            lineTo(x2, y2)
        }
        val gesture = GestureDescription.Builder()
            .addStroke(GestureDescription.StrokeDescription(path, 0, durationMs))
            .build()
        dispatchGesture(gesture, null, null)
    }

    private fun performKeyTap(key: String) {
        when (key.lowercase()) {
            "back" -> performGlobalAction(GLOBAL_ACTION_BACK)
            "home" -> performGlobalAction(GLOBAL_ACTION_HOME)
            "recents" -> performGlobalAction(GLOBAL_ACTION_RECENTS)
            "notifications" -> performGlobalAction(GLOBAL_ACTION_NOTIFICATIONS)
            "power_dialog" -> performGlobalAction(GLOBAL_ACTION_POWER_DIALOG)
            else -> android.util.Log.w("ControlService", "未知按键: $key")
        }
    }
}
