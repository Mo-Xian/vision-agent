package com.visionagent.client

import android.graphics.Bitmap
import okhttp3.*
import okio.ByteString
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit

/**
 * WebSocket 连接到 Vision Agent 中转服务（直连或公网中继）。
 *
 * 协议（与 PC 客户端 RemoteCaptureClient 完全一致）:
 *   客户端→中转: 二进制(8字节时间戳+JPEG), 文本JSON(meta/key_down/mouse_down等)
 *   中转→客户端: 文本JSON(cmd: key_tap/mouse_click 等控制指令)
 *
 * 中继模式: 连接后先发送 {"cmd":"join","room":"xxx","role":"client"}
 */
class HubConnection(
    private val serverUrl: String,
    private val fps: Int = 10,
    private val jpegQuality: Int = 70,
    private val roomId: String = "",
    private val relayToken: String = "",
    private val onLog: (String) -> Unit = {},
    private val onStatusChange: (Boolean) -> Unit = {},
    private val onControl: (JSONObject) -> Unit = {},
) {
    private val client = OkHttpClient.Builder()
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .build()

    private var ws: WebSocket? = null
    @Volatile var isConnected = false
        private set
    @Volatile private var running = false
    @Volatile private var paired = false

    private val startTime = System.currentTimeMillis()
    private var screenWidth = 0
    private var screenHeight = 0

    private val isRelayMode get() = roomId.isNotBlank()

    // 发送队列：避免在截屏线程直接调用 WebSocket
    private val sendQueue = LinkedBlockingQueue<Any>(60)
    private var senderThread: Thread? = null

    fun start(width: Int, height: Int) {
        screenWidth = width
        screenHeight = height
        running = true
        paired = false

        val request = Request.Builder().url(serverUrl).build()
        ws = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                onLog("[连接] 已连接: $serverUrl")

                if (isRelayMode) {
                    // 中继模式：先注册
                    val reg = JSONObject().apply {
                        put("cmd", "join")
                        put("room", roomId)
                        put("role", "client")
                        if (relayToken.isNotBlank()) put("token", relayToken)
                    }
                    webSocket.send(reg.toString())
                    onLog("[中继] 注册房间: $roomId")
                } else {
                    // 直连模式：直接开始
                    onConnected(webSocket)
                }
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                try {
                    val json = JSONObject(text)

                    // 处理中继事件
                    val event = json.optString("event", "")
                    if (event.isNotEmpty()) {
                        when (event) {
                            "joined" -> onLog("[中继] 已加入房间 $roomId")
                            "paired" -> {
                                onLog("[中继] 已配对，开始传输")
                                onConnected(webSocket)
                            }
                            "peer_left" -> {
                                onLog("[中继] Vision Agent 端已断开")
                                paired = false
                            }
                            "error" -> {
                                onLog("[中继] 错误: ${json.optString("msg")}")
                            }
                        }
                        return
                    }

                    // 直连模式的错误
                    if (json.optString("type") == "error") {
                        onLog("[连接] 服务端拒绝: ${json.optString("msg")}")
                        return
                    }

                    // 控制指令
                    if (json.has("cmd")) {
                        onControl(json)
                    }
                } catch (e: Exception) {
                    onLog("[连接] 消息解析错误: ${e.message}")
                }
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                val wasConnected = isConnected
                isConnected = false
                paired = false
                onStatusChange(false)
                if (wasConnected) {
                    onLog("[连接] 断开: ${t.message}")
                } else {
                    onLog("[连接] 连接失败: ${t.message}")
                }
                // 自动重连
                if (running) {
                    onLog("[连接] 5 秒后重连...")
                    Thread.sleep(5000)
                    if (running) {
                        start(screenWidth, screenHeight)
                    }
                }
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                isConnected = false
                paired = false
                onStatusChange(false)
                onLog("[连接] 已关闭: $reason")
            }
        })
    }

    private fun onConnected(webSocket: WebSocket) {
        isConnected = true
        paired = true
        onStatusChange(true)

        // 发送 meta
        val meta = JSONObject().apply {
            put("type", "meta")
            put("fps", fps)
            put("width", screenWidth)
            put("height", screenHeight)
            put("window", "android_screen")
        }
        webSocket.send(meta.toString())

        // 启动发送线程
        senderThread = Thread { senderLoop() }.apply {
            isDaemon = true
            start()
        }
    }

    fun stop() {
        running = false
        isConnected = false
        paired = false
        senderThread?.interrupt()
        ws?.close(1000, "客户端停止")
        ws = null
    }

    /**
     * 提交一帧画面到发送队列。
     */
    fun sendFrame(bitmap: Bitmap) {
        if (!isConnected || !paired) return

        val stream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, jpegQuality, stream)
        val jpegBytes = stream.toByteArray()

        val ts = (System.currentTimeMillis() - startTime) / 1000.0
        val buffer = ByteBuffer.allocate(8 + jpegBytes.size)
        buffer.putDouble(ts)
        buffer.put(jpegBytes)

        // 队列满则丢弃最旧帧
        if (sendQueue.remainingCapacity() == 0) {
            sendQueue.poll()
        }
        sendQueue.offer(buffer.array())
    }

    /**
     * 提交键鼠事件到发送队列。
     */
    fun sendEvent(event: JSONObject) {
        if (!isConnected || !paired) return
        sendQueue.offer(event.toString())
    }

    private fun senderLoop() {
        try {
            while (running && isConnected) {
                val item = sendQueue.poll(500, TimeUnit.MILLISECONDS) ?: continue
                if (!paired) continue
                val socket = ws ?: break
                when (item) {
                    is ByteArray -> socket.send(ByteString.of(*item))
                    is String -> socket.send(item)
                }
            }
        } catch (_: InterruptedException) {
            // 正常退出
        }
    }
}
