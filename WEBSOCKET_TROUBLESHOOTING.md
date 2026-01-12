# 🔧 WebSocket Connection Troubleshooting

## ❌ Problem: WebSocket Connection Failed

Jika Anda melihat error seperti ini di browser console:

```
WebSocket connection to 'wss://face.ahu.go.id/ws/auth/process-image/...' failed
```

**HTTPS API bekerja ✅** tapi **WebSocket gagal ❌**

## 🎯 Root Cause

WebSocket connections dari browser sering gagal karena:

### 1. **CORS Policy** (Paling Umum)
Browser memblokir WebSocket connection karena Cross-Origin Resource Sharing (CORS) policy. Server harus explicitly mengizinkan WebSocket connections dari origin Anda.

### 2. **WebSocket Origin Checking**
Server melakukan origin checking dan menolak connections dari browser.

### 3. **Firewall/Proxy**
Network firewall atau proxy blocking WebSocket upgrade requests.

### 4. **SSL/TLS Issues**
WSS (WebSocket Secure) memerlukan valid SSL certificate dan proper TLS configuration.

## ✅ Solutions

### Solution 1: Use Python CLI Client (RECOMMENDED) ⭐

**Ini adalah cara paling reliable!** Python client tidak terbatas oleh browser security policies.

```bash
# Menggunakan config profile
python test_websocket_auth.py --profile production enrollment 653384

# Atau dengan credentials langsung
python test_websocket_auth.py \
  frapi_YY7OEJn1FCyDoGiGLwiueTw79hkQWduGNy2L-XbsCB4 \
  _lwZfcqmdsi5PtRLjmOeTDgTxP5JaRAN3r4i6IpCSOC6ndL536sO9ZuFVjbgLshbmuNKmBButy_wZgdyXEw-DA \
  https://face.ahu.go.id \
  enrollment \
  653384
```

**Keuntungan:**
- ✅ Tidak ada CORS issues
- ✅ Tidak ada browser limitations
- ✅ Reliable WebSocket connection
- ✅ Lebih mudah untuk automation
- ✅ Better error handling

### Solution 2: Server-Side Configuration

Jika Anda mengontrol server, tambahkan CORS headers untuk WebSocket:

**Django Channels (daphne/uvicorn):**

```python
# settings.py
CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels_redis.core.RedisChannelLayer',
        'CONFIG': {
            "hosts": [('127.0.0.1', 6379)],
        },
    },
}

# Allowed hosts
ALLOWED_HOSTS = ['*']  # or specific domains

# CORS settings
CORS_ALLOW_ALL_ORIGINS = True  # Development only
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOWED_ORIGINS = [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]

# WebSocket CORS
CORS_ALLOW_HEADERS = [
    'accept',
    'accept-encoding',
    'authorization',
    'content-type',
    'origin',
    'user-agent',
    'x-csrftoken',
    'x-requested-with',
]
```

**Nginx Configuration:**

```nginx
location /ws/ {
    proxy_pass http://backend;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    
    # WebSocket timeout settings
    proxy_read_timeout 86400;
    proxy_send_timeout 86400;
    
    # CORS headers for WebSocket
    add_header 'Access-Control-Allow-Origin' '*' always;
    add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS' always;
    add_header 'Access-Control-Allow-Headers' 'Origin, Content-Type, Accept, Authorization' always;
}
```

### Solution 3: Browser Extensions (Temporary Workaround)

**⚠️ Only for testing!**

Install browser extension untuk disable CORS:
- Chrome: "Allow CORS: Access-Control-Allow-Origin"
- Firefox: "CORS Everywhere"

**WARNING:** Ini hanya untuk development/testing. Jangan digunakan untuk production!

### Solution 4: Local Proxy

Buat local proxy server yang meneruskan WebSocket connections:

```python
# proxy_server.py
import asyncio
import websockets
from aiohttp import web, ClientSession

async def websocket_proxy(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    # Connect to actual server
    target_url = 'wss://face.ahu.go.id/ws/auth/process-image/...'
    async with websockets.connect(target_url) as target_ws:
        # Forward messages both ways
        async def forward_to_server():
            async for msg in ws:
                await target_ws.send(msg.data)
        
        async def forward_to_client():
            async for msg in target_ws:
                await ws.send_str(msg)
        
        await asyncio.gather(
            forward_to_server(),
            forward_to_client()
        )
    
    return ws

app = web.Application()
app.router.add_get('/ws/{path:.*}', websocket_proxy)
web.run_app(app, host='127.0.0.1', port=9000)
```

Kemudian connect ke `ws://localhost:9000/...` dari browser.

## 🔍 Diagnostic Steps

### Step 1: Check Browser Console

Buka Browser DevTools (F12) dan check:

1. **Console tab** - Lihat error messages
2. **Network tab** - Filter "WS" untuk WebSocket connections
3. Check response headers dan status codes

### Step 2: Check WebSocket URL

Pastikan URL valid:
```javascript
// Good
wss://face.ahu.go.id/ws/auth/process-image/sess_xxx/

// Bad
ws://face.ahu.go.id/ws/auth/process-image/sess_xxx/  // Should be wss:// not ws://
wss://face.ahu.go.id/ws/auth/process-image/sess_xxx  // Missing trailing slash
```

### Step 3: Test dengan curl/wscat

Test WebSocket connection dari command line:

```bash
# Install wscat
npm install -g wscat

# Test connection
wscat -c "wss://face.ahu.go.id/ws/auth/process-image/sess_xxx/"
```

Jika ini berhasil tapi browser gagal, berarti CORS issue.

### Step 4: Check Server Logs

Check server logs untuk melihat apakah connection request sampai ke server:

```bash
# Django
tail -f /path/to/logs/django.log

# Nginx
tail -f /var/log/nginx/error.log
```

## 📊 Comparison: Web vs CLI

| Feature | Web Client | Python CLI |
|---------|-----------|-----------|
| Setup | Easy (browser only) | Need Python + dependencies |
| CORS Issues | ❌ Yes | ✅ No |
| WebSocket | ❌ May fail | ✅ Always works |
| Visual Feedback | ✅ Yes | ⚠️ Limited |
| Automation | ❌ Difficult | ✅ Easy |
| Production Ready | ❌ No | ✅ Yes |
| Debugging | ⚠️ Limited | ✅ Full control |

## 🎯 Recommended Approach

### For Development/Testing:
```bash
# Use CLI client
python test_websocket_auth.py --profile production enrollment 653384
```

### For Demo/Presentation:
```bash
# Use web client on local development server
# Edit server to allow CORS from localhost
```

### For Production:
```bash
# Always use CLI client
# Or implement proper WebSocket proxy
```

## 🔐 Security Considerations

WebSocket CORS restrictions exist for security reasons. Bypassing them should only be done:

1. **Development environment only**
2. **Trusted local network**
3. **With proper understanding of risks**

**Never in production:**
- Don't disable CORS globally
- Don't use browser extensions
- Don't expose credentials

## 📞 Need Help?

Jika masih mengalami issues:

1. **Check server configuration** - Pastikan WebSocket properly configured
2. **Use CLI client** - Ini always works
3. **Contact server admin** - Mungkin perlu CORS whitelist
4. **Check firewall** - Pastikan WebSocket tidak di-block

## 💡 Quick Fix Summary

**Problem:** WebSocket connection fails from browser
**Quick Solution:** Use Python CLI instead

```bash
python test_websocket_auth.py --profile production enrollment 653384
```

✅ This bypasses all browser limitations and works reliably!
