# OIDC Testing Guide

Panduan lengkap untuk testing OAuth 2.0 / OpenID Connect integration dengan Face Recognition.

## Prerequisites

1. Django server running
2. Minimal satu user dengan wajah terdaftar
3. OAuth Client sudah dibuat

## Quick Start

### 1. Buat OAuth Client

```bash
cd face_recognition_app
python manage.py create_oidc_client
```

Output contoh:
```
Client ID: oidc_abc123...
Client Secret: secret_xyz789...
```

### 2. Test Discovery & JWKS

```bash
# Test discovery endpoint
curl https://face.ahu.go.id/.well-known/openid-configuration

# Test JWKS
curl https://face.ahu.go.id/.well-known/jwks.json
```

### 3. Test via Browser (Demo Page)

1. Buka browser: `https://face.ahu.go.id/oauth/demo`
2. Masukkan Client ID dan Secret
3. Klik "Test Discovery" untuk verifikasi
4. Klik "Start Login" untuk test face authentication
5. Setelah berhasil, token akan tampil di halaman

### 4. Test via Keycloak Simulator

```bash
# Interactive mode (opens browser)
python keycloak_simulator.py \
    --client-id oidc_xxx \
    --client-secret secret_xxx

# Manual mode (for servers without browser)
python keycloak_simulator.py \
    --client-id oidc_xxx \
    --client-secret secret_xxx \
    --manual

# Test only (no login)
python keycloak_simulator.py \
    --client-id oidc_xxx \
    --client-secret secret_xxx \
    --test-only
```

### 5. Run End-to-End Tests

```bash
# With existing client
python test_oidc_e2e.py \
    --client-id oidc_xxx \
    --client-secret secret_xxx

# Create test client automatically
python test_oidc_e2e.py

# With cleanup after test
python test_oidc_e2e.py --cleanup
```

### 6. Test OIDC Flow Script

```bash
python test_oidc_flow.py \
    --client-id oidc_xxx \
    --client-secret secret_xxx
```

## Test Scenarios

### Scenario 1: Basic Authorization Code Flow

1. **Authorize Request**
   ```
   GET /oauth/authorize?
     client_id=oidc_xxx&
     redirect_uri=http://localhost:8080/callback&
     response_type=code&
     scope=openid profile email&
     state=random_state&
     nonce=random_nonce
   ```

2. **Face Authentication**
   - User diarahkan ke halaman face login
   - Kamera menyala untuk liveness detection
   - Face recognition dijalankan

3. **Authorization Code**
   ```
   GET http://localhost:8080/callback?
     code=auth_code_xxx&
     state=random_state
   ```

4. **Token Exchange**
   ```bash
   POST /oauth/token
   Content-Type: application/x-www-form-urlencoded

   grant_type=authorization_code&
   code=auth_code_xxx&
   redirect_uri=http://localhost:8080/callback&
   client_id=oidc_xxx&
   client_secret=secret_xxx
   ```

5. **Response**
   ```json
   {
     "access_token": "eyJhbGciOiJSUzI1Ni...",
     "token_type": "Bearer",
     "expires_in": 3600,
     "id_token": "eyJhbGciOiJSUzI1Ni...",
     "refresh_token": "refresh_xxx..."
   }
   ```

### Scenario 2: With PKCE

```bash
# Generate code_verifier
code_verifier=$(openssl rand -base64 64 | tr -d '=' | tr '/+' '_-' | cut -c1-128)

# Generate code_challenge
code_challenge=$(echo -n $code_verifier | openssl dgst -sha256 -binary | base64 | tr -d '=' | tr '/+' '_-')

# Authorization request with PKCE
curl "https://face.ahu.go.id/oauth/authorize?\
client_id=oidc_xxx&\
redirect_uri=http://localhost:8080/callback&\
response_type=code&\
scope=openid profile email&\
code_challenge=$code_challenge&\
code_challenge_method=S256"

# Token exchange with code_verifier
curl -X POST https://face.ahu.go.id/oauth/token \
  -d "grant_type=authorization_code" \
  -d "code=xxx" \
  -d "redirect_uri=http://localhost:8080/callback" \
  -d "client_id=oidc_xxx" \
  -d "code_verifier=$code_verifier"
```

### Scenario 3: UserInfo

```bash
curl -H "Authorization: Bearer ACCESS_TOKEN" \
  https://face.ahu.go.id/oauth/userinfo
```

Response:
```json
{
  "sub": "uuid-user-id",
  "name": "John Doe",
  "email": "john@example.com",
  "email_verified": true,
  "face_verified": true,
  "liveness_verified": true
}
```

### Scenario 4: Token Refresh

```bash
curl -X POST https://face.ahu.go.id/oauth/token \
  -d "grant_type=refresh_token" \
  -d "refresh_token=REFRESH_TOKEN" \
  -d "client_id=oidc_xxx" \
  -d "client_secret=secret_xxx"
```

### Scenario 5: Token Introspection

```bash
curl -X POST https://face.ahu.go.id/oauth/introspect \
  -d "token=ACCESS_TOKEN" \
  -d "client_id=oidc_xxx" \
  -d "client_secret=secret_xxx"
```

### Scenario 6: Token Revocation

```bash
curl -X POST https://face.ahu.go.id/oauth/revoke \
  -d "token=ACCESS_TOKEN" \
  -d "token_type_hint=access_token" \
  -d "client_id=oidc_xxx" \
  -d "client_secret=secret_xxx"
```

## Keycloak Integration

### 1. Create Identity Provider in Keycloak

1. Login ke Keycloak Admin Console
2. Pilih realm Anda
3. Navigate: Identity Providers → Add Provider → OpenID Connect v1.0

### 2. Configure Provider

| Field | Value |
|-------|-------|
| Alias | face-recognition |
| Display Name | Face Recognition Login |
| Discovery URL | https://face.ahu.go.id/.well-known/openid-configuration |
| Client ID | (dari create_oidc_client) |
| Client Secret | (dari create_oidc_client) |
| Default Scopes | openid profile email face_auth |

### 3. Advanced Settings

- Backchannel Logout: Enabled
- User Info URL: https://face.ahu.go.id/oauth/userinfo
- PKCE Method: S256

### 4. Mapper Configuration

Tambahkan mappers untuk menyinkronkan atribut:

**Face Verified Mapper:**
- Type: Attribute Importer
- Claim: face_verified
- User Attribute Name: face_verified

**Liveness Mapper:**
- Type: Attribute Importer
- Claim: liveness_verified  
- User Attribute Name: liveness_verified

## Troubleshooting

### Error: invalid_client

**Penyebab:** Client ID atau secret salah
**Solusi:** Verifikasi credentials di admin atau buat client baru

### Error: invalid_redirect_uri

**Penyebab:** Redirect URI tidak terdaftar di client
**Solusi:** Tambahkan redirect URI ke OAuth Client di admin

### Error: access_denied

**Penyebab:** Face authentication gagal atau user menolak consent
**Solusi:** Pastikan wajah user sudah terdaftar dan pencahayaan cukup

### Error: invalid_grant

**Penyebab:** Authorization code sudah digunakan atau expired
**Solusi:** Minta authorization code baru

### JWKS Error

**Penyebab:** RSA key belum di-generate
**Solusi:** 
```bash
# Keys akan di-generate otomatis saat pertama akses
curl https://face.ahu.go.id/.well-known/jwks.json
```

## Test Results Template

```
============================
OIDC Test Results
============================
Date: 2024-XX-XX
Environment: Production/Staging

Discovery Endpoint: ✅ PASS
JWKS Endpoint: ✅ PASS
Authorization: ✅ PASS
Token Exchange: ✅ PASS
UserInfo: ✅ PASS
Token Refresh: ✅ PASS
Introspection: ✅ PASS
Revocation: ✅ PASS

Total: 8/8 (100%)
============================
```
