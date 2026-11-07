package io.github.felipemagr.selecta

import android.app.Activity
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
import android.view.KeyEvent
import android.view.WindowManager
import android.webkit.JavascriptInterface
import android.webkit.WebResourceError
import android.webkit.WebResourceRequest
import android.webkit.WebView
import android.webkit.WebViewClient

/** The TV screen: Selecta's /tv page, full screen, with sound that needs no tap. */
class MainActivity : Activity() {
    private lateinit var web: WebView
    private val prefs by lazy { getSharedPreferences("selecta", MODE_PRIVATE) }
    private val server: String? get() = prefs.getString("server", null)

    override fun onCreate(state: Bundle?) {
        super.onCreate(state)
        web = WebView(this).apply {
            setBackgroundColor(Color.parseColor("#0b0a09"))
            settings.javaScriptEnabled = true
            settings.domStorageEnabled = true
            settings.mediaPlaybackRequiresUserGesture = false
            addJavascriptInterface(Bridge(), "SelectaTV")
            webViewClient = Client()
        }
        setContentView(web)
        server?.let(::open) ?: setup(null)
    }

    private fun open(server: String) = web.loadUrl("$server/tv")

    private fun setup(error: String?) {
        val query = Uri.Builder()
            .appendQueryParameter("server", server ?: "")
            .appendQueryParameter("error", error ?: "")
            .build().encodedQuery
        web.loadUrl("$SETUP?$query")
    }

    override fun onKeyDown(code: Int, event: KeyEvent): Boolean {
        if (code == KeyEvent.KEYCODE_BACK) event.startTracking()
        return super.onKeyDown(code, event)
    }

    // Holding Back changes the server; a short press leaves the app.
    override fun onKeyLongPress(code: Int, event: KeyEvent): Boolean {
        if (code != KeyEvent.KEYCODE_BACK) return super.onKeyLongPress(code, event)
        setup(null)
        return true
    }

    override fun onDestroy() {
        web.destroy()
        super.onDestroy()
    }

    private inner class Client : WebViewClient() {
        override fun onReceivedError(view: WebView, request: WebResourceRequest, error: WebResourceError) {
            if (request.isForMainFrame && !request.url.toString().startsWith(SETUP)) {
                setup("Can't reach ${request.url.host}: ${error.description}")
            }
        }

        // Only the setup page and the chosen server open here.
        override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest): Boolean {
            val url = request.url.toString()
            return !(url.startsWith(SETUP) || server?.let { url.startsWith("$it/") } == true)
        }
    }

    private inner class Bridge {
        @JavascriptInterface
        fun keepAwake(on: Boolean) = runOnUiThread {
            if (on) window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            else window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        }

        @JavascriptInterface
        fun connect(address: String) = runOnUiThread {
            // Only the setup page may change the server.
            if (web.url?.startsWith(SETUP) != true) return@runOnUiThread
            val server = address.trim().trimEnd('/')
            if (!server.startsWith("https://")) return@runOnUiThread setup("The address must start with https://")
            prefs.edit().putString("server", server).apply()
            open(server)
        }
    }

    private companion object {
        const val SETUP = "file:///android_asset/setup.html"
    }
}
