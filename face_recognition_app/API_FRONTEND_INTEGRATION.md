# 📘 Integrasi Frontend dengan Face Recognition API (Ringkasan Utama)

Dokumentasi ini adalah ringkasan eksekutif untuk integrasi frontend. Detail lengkap tiap domain (Authentication, Enrollment, Identification/Verification, Analytics) dipisah dalam dokumen khusus agar modular dan mudah dirawat.

## 🔗 Dokumen Detail


| Domain | Deskripsi | File |
|--------|-----------|------|
| Authentication & Session | Registrasi, login password, login wajah (privat & publik), refresh token, manajemen sesi | `docs/authentication.md` |
| Enrollment + Liveness | Proses pembuatan template wajah via WebRTC + pengiriman frame base64 | `docs/enrollment.md` |
| Identification & Verification | Alur autentikasi wajah (matching), liveness, mode verification terhadap user tertentu | `docs/identification_verification.md` |
| Analytics & Monitoring | Log autentikasi, security alert, dashboard statistik, KPI | `docs/analytics.md` |
| WebRTC Implementation | Implementasi WebSocket/WebRTC paralel untuk enrollment & auth | `docs/webrtc.md` |

> Semua endpoint sudah tersedia di Swagger (`/api/docs/`) & Redoc (`/api/redoc/`) – namun dokumen ini memberi konteks arsitektur, flow, dan contoh implementasi frontend.

## ✅ Prinsip Integrasi Frontend

1. Gunakan JWT (access + refresh) untuk endpoint yang memerlukan autentikasi.
2. Proses berbasis video (enrollment & authentication wajah) terdiri dari 2 layer: (a) WebRTC session (opsional utk real-time) dan (b) HTTP endpoint pengiriman frame base64 bertahap.
3. Liveness minimal: jumlah frame + gerakan/blink (dinamis dari server). Frontend cukup terus mengirim frame sampai server menyatakan sukses/final.
4. Mode Face Login Publik tidak memerlukan JWT di awal – cukup email + device info → jika match sukses, server mengembalikan token.
5. Simpan `session_token` dengan scoping ketat (per tab / per proses) dan jangan reuse setelah final.

## 🧩 High-Level Arsitektur

```mermaid
flowchart LR
	A[User Kamera Browser] --> B[Frontend (Vue)]
	B --> C{Jenis Proses}
	C -->|Enrollment| E[/POST /api/enrollment/create/ \nloop POST /api/enrollment/process-frame/]
	C -->|Face Login| F[/POST /api/auth/face/public/create/ \nloop POST /api/auth/face/process-frame/]
	C -->|Auth (JWT)| G[/POST /api/auth/face/create/ \nloop POST /api/auth/face/process-frame/]
	C -->|Password Login| H[/POST /api/auth/token/]
	E --> I[(Engine + DB)]
	F --> I
	G --> I
	I --> J[(Analytics & Security Logs)]
	J --> K[/GET /api/analytics/.../]
```

## 🧪 Minimal Frontend Loop (Pseudo)

```javascript
// 1. Create session (auth / enrollment)
const { session_token } = await api.post('/api/auth/face/create/', { session_type: 'authentication', device_info });

// 2. Capture frame (canvas -> toDataURL)
const frameData = canvas.toDataURL('image/jpeg', 0.8);

// 3. Send frame iteratively
const res = await api.post('/api/auth/face/process-frame/', { session_token, frame_data: frameData });
if (res.data.requires_more_frames) continueLoop(); else finalize();
```

## 🛡️ Security & Best Practice

- Batasi resolusi frame sebelum kirim (misal resize 320x240) untuk latency rendah.
- Jangan cache frame mentah di IndexedDB/localStorage.
- Pakai `device_info` konsisten (UUID per instalasi / localStorage).
- Tangani status `requires_new_session` untuk mengulang otomatis dengan UX halus.

 
## 📂 Lanjutkan Membaca

Pergi ke dokumen domain spesifik di folder `docs/` untuk detail end-to-end.

---

Terakhir diperbarui: (auto) *silakan update manual bila perlu.*

