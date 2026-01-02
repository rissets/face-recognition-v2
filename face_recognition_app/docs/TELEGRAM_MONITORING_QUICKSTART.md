# Panduan Singkat Telegram Error Monitoring

## 🚀 Quick Start

### 1. Buat Bot Telegram
```bash
1. Cari @BotFather di Telegram
2. Kirim /newbot
3. Ikuti instruksi
4. Simpan Bot Token yang diberikan
```

### 2. Dapatkan Chat ID
```bash
1. Cari @userinfobot di Telegram
2. Kirim pesan apa saja
3. Simpan Chat ID yang diberikan
```

### 3. Konfigurasi .env
```bash
TELEGRAM_ERROR_LOGGING_ENABLED=True
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=987654321
ENVIRONMENT=production
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Test Notifikasi
```bash
python manage.py test_telegram_monitoring
```

## 📱 Notifikasi Otomatis

Sistem akan otomatis mengirim notifikasi untuk:
- ❌ Semua unhandled exceptions
- 🔴 HTTP 5xx errors
- 🛡️ Security threats (SQL injection, XSS, etc.)
- 💾 Database errors
- ⚙️ Celery task failures

## 💻 Manual Logging

```python
from core.telegram_logger import telegram_logger

# Log error
telegram_logger.log_critical_error(
    message="Something went wrong",
    exception=exception,
    additional_context={'user_id': 123}
)
```

## 🔍 Testing

```bash
# Test semua tipe notifikasi
python manage.py test_telegram_monitoring --type=all

# Test specific type
python manage.py test_telegram_monitoring --type=critical
python manage.py test_telegram_monitoring --type=security
```

## 📖 Dokumentasi Lengkap

Lihat [TELEGRAM_MONITORING.md](./TELEGRAM_MONITORING.md) untuk dokumentasi lengkap.

## ⚠️ Troubleshooting

### Bot tidak mengirim notifikasi?
1. Pastikan `TELEGRAM_ERROR_LOGGING_ENABLED=True`
2. Cek Bot Token dan Chat ID benar
3. Kirim pesan ke bot terlebih dahulu
4. Test dengan: `python manage.py test_telegram_monitoring`

### Error "Unauthorized"?
- Bot Token salah atau expired
- Regenerate token dari @BotFather

### Error "Chat not found"?
- Chat ID salah
- Bot belum pernah menerima pesan dari chat tersebut

## 🔐 Security

Jangan commit Bot Token ke repository!
```bash
# Tambahkan ke .gitignore
.env
.env.local
```

## 📊 Contoh Notifikasi

```
🚨 CRITICAL ERROR
Environment: production
Time: 2026-01-02 14:30:45 UTC

Message:
Database connection timeout

Exception Type: OperationalError
Exception Details:
could not connect to server

Request Info:
• Method: POST
• Path: /api/auth/face-recognition/
• IP: 192.168.1.100
```
