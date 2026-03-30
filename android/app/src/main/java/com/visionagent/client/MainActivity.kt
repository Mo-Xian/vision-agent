package com.visionagent.client

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.button.MaterialButton
import com.google.android.material.textfield.TextInputEditText

/**
 * 主界面：输入服务器地址 → 授权截屏 → 启动采集服务。
 */
class MainActivity : AppCompatActivity() {

    private lateinit var etServerUrl: TextInputEditText
    private lateinit var etRoomId: TextInputEditText
    private lateinit var etRelayToken: TextInputEditText
    private lateinit var etFps: TextInputEditText
    private lateinit var etQuality: TextInputEditText
    private lateinit var btnConnect: MaterialButton
    private lateinit var btnAccessibility: MaterialButton
    private lateinit var tvStatus: TextView
    private lateinit var tvLog: TextView
    private lateinit var scrollLog: ScrollView

    private var isCapturing = false

    // MediaProjection 授权回调
    private val projectionLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK && result.data != null) {
            startCaptureService(result.resultCode, result.data!!)
        } else {
            appendLog("[错误] 用户拒绝了截屏授权")
            updateStatus("截屏授权被拒绝", false)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        etServerUrl = findViewById(R.id.etServerUrl)
        etRoomId = findViewById(R.id.etRoomId)
        etRelayToken = findViewById(R.id.etRelayToken)
        etFps = findViewById(R.id.etFps)
        etQuality = findViewById(R.id.etQuality)
        btnConnect = findViewById(R.id.btnConnect)
        btnAccessibility = findViewById(R.id.btnAccessibility)
        tvStatus = findViewById(R.id.tvStatus)
        tvLog = findViewById(R.id.tvLog)
        scrollLog = tvLog.parent as ScrollView

        // 恢复上次的设置
        val prefs = getSharedPreferences("settings", Context.MODE_PRIVATE)
        prefs.getString("server_url", null)?.let { etServerUrl.setText(it) }
        prefs.getString("room_id", null)?.let { etRoomId.setText(it) }
        prefs.getString("relay_token", null)?.let { etRelayToken.setText(it) }
        prefs.getString("fps", null)?.let { etFps.setText(it) }
        prefs.getString("quality", null)?.let { etQuality.setText(it) }

        btnConnect.setOnClickListener {
            if (isCapturing) {
                stopCapture()
            } else {
                requestCapture()
            }
        }

        btnAccessibility.setOnClickListener {
            openAccessibilitySettings()
        }

        updateAccessibilityButton()
    }

    override fun onResume() {
        super.onResume()
        updateAccessibilityButton()
    }

    private fun requestCapture() {
        val url = etServerUrl.text?.toString()?.trim() ?: ""
        if (url.isEmpty()) {
            Toast.makeText(this, "请输入中转服务地址", Toast.LENGTH_SHORT).show()
            return
        }

        // 保存设置
        getSharedPreferences("settings", Context.MODE_PRIVATE).edit().apply {
            putString("server_url", url)
            putString("room_id", etRoomId.text?.toString())
            putString("relay_token", etRelayToken.text?.toString())
            putString("fps", etFps.text?.toString())
            putString("quality", etQuality.text?.toString())
            apply()
        }

        // 请求 MediaProjection 授权
        val projManager = getSystemService(MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
        projectionLauncher.launch(projManager.createScreenCaptureIntent())
        appendLog("[系统] 请求截屏授权...")
    }

    private fun startCaptureService(resultCode: Int, resultData: Intent) {
        val url = etServerUrl.text?.toString()?.trim() ?: return
        val fps = etFps.text?.toString()?.toIntOrNull() ?: 10
        val quality = etQuality.text?.toString()?.toIntOrNull() ?: 70
        val roomId = etRoomId.text?.toString()?.trim() ?: ""
        val relayToken = etRelayToken.text?.toString()?.trim() ?: ""

        val intent = Intent(this, CaptureService::class.java).apply {
            putExtra(CaptureService.EXTRA_RESULT_CODE, resultCode)
            putExtra(CaptureService.EXTRA_RESULT_DATA, resultData)
            putExtra(CaptureService.EXTRA_SERVER_URL, url)
            putExtra(CaptureService.EXTRA_FPS, fps)
            putExtra(CaptureService.EXTRA_QUALITY, quality)
            putExtra(CaptureService.EXTRA_ROOM_ID, roomId)
            putExtra(CaptureService.EXTRA_RELAY_TOKEN, relayToken)
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(intent)
        } else {
            startService(intent)
        }

        isCapturing = true
        btnConnect.text = "停止采集"
        updateStatus("采集中 → $url", true)
        val modeStr = if (roomId.isNotEmpty()) "中继 房间=$roomId" else "直连"
        appendLog("[采集] 服务已启动: $url ($modeStr, FPS=$fps, Q=$quality)")
        setInputEnabled(false)
    }

    private fun stopCapture() {
        CaptureService.instance?.stopCapture()
        isCapturing = false
        btnConnect.text = "开始采集"
        updateStatus("已停止", false)
        appendLog("[采集] 服务已停止")
        setInputEnabled(true)
    }

    private fun openAccessibilitySettings() {
        val intent = Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)
        startActivity(intent)
        Toast.makeText(this, "请在列表中找到 \"Vision Agent\" 并开启", Toast.LENGTH_LONG).show()
    }

    private fun updateAccessibilityButton() {
        if (ControlService.isEnabled()) {
            btnAccessibility.text = "无障碍服务已开启 ✓"
            btnAccessibility.isEnabled = false
        } else {
            btnAccessibility.text = "开启无障碍服务（Agent 控制需要）"
            btnAccessibility.isEnabled = true
        }
    }

    private fun updateStatus(text: String, active: Boolean) {
        tvStatus.text = text
        tvStatus.setTextColor(
            if (active) 0xFF00B894.toInt() else 0xFF999999.toInt()
        )
    }

    private fun appendLog(msg: String) {
        runOnUiThread {
            tvLog.append("$msg\n")
            scrollLog.post { scrollLog.fullScroll(ScrollView.FOCUS_DOWN) }
        }
    }

    private fun setInputEnabled(enabled: Boolean) {
        etServerUrl.isEnabled = enabled
        etRoomId.isEnabled = enabled
        etRelayToken.isEnabled = enabled
        etFps.isEnabled = enabled
        etQuality.isEnabled = enabled
    }
}
