# Integrasi Keycloak dengan Face Recognition OIDC Provider

Panduan lengkap untuk mengintegrasikan Face Recognition sebagai Identity Provider di Keycloak.

## Gambaran Umum

Face Recognition App sekarang mendukung OAuth 2.0 / OpenID Connect 1.0 yang memungkinkan:
- Digunakan sebagai External Identity Provider di Keycloak
- Autentikasi biometrik wajah untuk login
- Single Sign-On (SSO) dengan verifikasi wajah
- Multi-factor authentication (MFA) berbasis wajah

## Endpoints OIDC

| Endpoint | URL | Deskripsi |
|----------|-----|-----------|
| Discovery | `/.well-known/openid-configuration` | OpenID Connect Discovery |
| JWKS | `/.well-known/jwks.json` | JSON Web Key Set |
| Authorization | `/oauth/authorize` | Authorization endpoint |
| Token | `/oauth/token` | Token endpoint |
| UserInfo | `/oauth/userinfo` | User information endpoint |
| Revocation | `/oauth/revoke` | Token revocation |
| Introspection | `/oauth/introspect` | Token introspection |
| End Session | `/oauth/logout` | Logout endpoint |

## Konfigurasi di Keycloak

### Langkah 1: Tambahkan Identity Provider

1. Login ke Keycloak Admin Console
2. Pilih Realm yang akan dikonfigurasi
3. Buka **Identity Providers** di sidebar
4. Klik **Add provider** → **OpenID Connect v1.0**

### Langkah 2: Konfigurasi Identity Provider

Isi form dengan detail berikut:

```
Alias: face-recognition
Display Name: Face Recognition Login

# Discovery Settings
Use discovery endpoint: ON
Discovery endpoint: https://face.ahu.go.id/.well-known/openid-configuration

# Atau konfigurasi manual:
Authorization URL: https://face.ahu.go.id/oauth/authorize
Token URL: https://face.ahu.go.id/oauth/token
Logout URL: https://face.ahu.go.id/oauth/logout
User Info URL: https://face.ahu.go.id/oauth/userinfo
JWKS URL: https://face.ahu.go.id/.well-known/jwks.json

# Client Configuration
Client ID: [dari Face Recognition Admin]
Client Secret: [dari Face Recognition Admin]
Client Authentication: Client secret sent as post

# Scopes
Default Scopes: openid profile email face_auth

# Other Settings
Validate Signatures: ON
Use JWKS URL: ON
PKCE Method: S256
```

### Langkah 3: Buat OAuth Client di Face Recognition

1. Akses Django Admin di `https://face.ahu.go.id/admin/`
2. Buka **Auth Service** → **OAuth Clients**
3. Klik **Add OAuth Client**
4. Isi form:

```
Name: Keycloak Production
Description: Keycloak Identity Broker

Client Type: Confidential

Redirect URIs:
https://keycloak.example.com/realms/myrealm/broker/face-recognition/endpoint
https://keycloak.example.com/realms/myrealm/broker/face-recognition/endpoint/*

Grant Types: ["authorization_code", "refresh_token"]
Response Types: ["code"]
Allowed Scopes: ["openid", "profile", "email", "face_auth"]

Token Settings:
- Access Token Lifetime: 3600
- Refresh Token Lifetime: 86400
- ID Token Lifetime: 3600

Security:
- Require PKCE: ON
- Require Consent: ON

Face Authentication:
- Require Face Auth: ON
- Require Liveness: ON
- Min Confidence Score: 0.85
```

5. Simpan dan catat **Client ID** dan **Client Secret**

### Langkah 4: Mapper Konfigurasi

Buat mappers di Keycloak untuk memetakan claims:

#### Mapper 1: Username
```
Name: face-username
Mapper Type: Username Template Importer
Template: ${CLAIM.sub}
Target: LOCAL
```

#### Mapper 2: First Name
```
Name: face-firstname
Mapper Type: Attribute Importer
Claim: given_name
User Attribute Name: firstName
```

#### Mapper 3: Last Name
```
Name: face-lastname
Mapper Type: Attribute Importer
Claim: family_name
User Attribute Name: lastName
```

#### Mapper 4: Email
```
Name: face-email
Mapper Type: Attribute Importer
Claim: email
User Attribute Name: email
```

#### Mapper 5: Face Verified (Custom)
```
Name: face-verified
Mapper Type: Attribute Importer
Claim: face_verified
User Attribute Name: faceVerified
```

## Flow Autentikasi

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────────┐
│   User      │     │   Keycloak   │     │  Face Recognition   │
└──────┬──────┘     └──────┬───────┘     └──────────┬──────────┘
       │                   │                        │
       │ 1. Login Request  │                        │
       │──────────────────>│                        │
       │                   │                        │
       │                   │ 2. Redirect to Face    │
       │                   │    Recognition         │
       │<──────────────────│──────────────────────>│
       │                   │                        │
       │ 3. Face Auth UI   │                        │
       │<──────────────────────────────────────────│
       │                   │                        │
       │ 4. Camera Stream  │                        │
       │   (WebSocket)     │                        │
       │───────────────────────────────────────────>│
       │                   │                        │
       │ 5. Face Verified  │                        │
       │<──────────────────────────────────────────│
       │                   │                        │
       │                   │ 6. Auth Code           │
       │──────────────────>│<──────────────────────│
       │                   │                        │
       │                   │ 7. Exchange for Tokens │
       │                   │──────────────────────>│
       │                   │                        │
       │                   │ 8. ID Token + Access   │
       │                   │<──────────────────────│
       │                   │                        │
       │ 9. Session Created│                        │
       │<──────────────────│                        │
       │                   │                        │
```

## Scopes dan Claims

### Scopes yang Didukung

| Scope | Deskripsi |
|-------|-----------|
| `openid` | Required untuk OIDC |
| `profile` | Nama, foto profil |
| `email` | Alamat email |
| `face_auth` | Informasi autentikasi wajah |
| `offline_access` | Refresh token |

### Claims yang Tersedia

```json
{
  "sub": "user-uuid",
  "iss": "https://face.ahu.go.id",
  "aud": "client_id",
  "exp": 1704825600,
  "iat": 1704822000,
  "auth_time": 1704822000,
  "nonce": "...",
  "at_hash": "...",
  
  // Profile claims
  "name": "John Doe",
  "given_name": "John",
  "family_name": "Doe",
  "picture": "https://...",
  
  // Email claims
  "email": "john@example.com",
  "email_verified": true,
  
  // Face auth claims
  "face_verified": true,
  "face_confidence": 0.95,
  "liveness_verified": true
}
```

## Konfigurasi Lanjutan

### Mengaktifkan Face Auth sebagai Step-up Authentication

Di Keycloak, Anda bisa membuat Authentication Flow khusus:

1. Buka **Authentication** → **Flows**
2. Copy **Browser** flow
3. Tambahkan step **Identity Provider Redirector**
4. Set default provider ke `face-recognition`

### Conditional Face Auth

Untuk aplikasi tertentu yang membutuhkan verifikasi wajah:

```json
{
  "clientId": "sensitive-app",
  "defaultClientScopes": ["openid", "face_auth"],
  "authenticationFlowBindingOverrides": {
    "browser": "face-auth-flow"
  }
}
```

## Troubleshooting

### Error: Invalid redirect_uri

Pastikan redirect URI di OAuth Client Face Recognition sesuai dengan:
```
https://keycloak.domain.com/realms/{realm}/broker/face-recognition/endpoint
```

### Error: PKCE verification failed

Pastikan Keycloak menggunakan `S256` sebagai PKCE method.

### Face Auth tidak muncul

1. Cek apakah user sudah enrolled di Face Recognition
2. Pastikan `require_face_auth` aktif di OAuth Client
3. Periksa koneksi WebSocket ke face-auth endpoint

### Token expired

Sesuaikan lifetime di OAuth Client:
- Access Token: minimal 3600 detik
- Refresh Token: minimal 86400 detik

## Security Best Practices

1. **Selalu gunakan HTTPS** untuk semua endpoint
2. **Aktifkan PKCE** untuk mencegah authorization code interception
3. **Validasi redirect_uri** secara strict
4. **Rotate client secrets** secara berkala
5. **Monitor failed authentication** attempts
6. **Set confidence score** sesuai kebutuhan keamanan

## API Testing

### Test Discovery Endpoint
```bash
curl https://face.ahu.go.id/.well-known/openid-configuration
```

### Test JWKS Endpoint
```bash
curl https://face.ahu.go.id/.well-known/jwks.json
```

### Test Token Endpoint
```bash
curl -X POST https://face.ahu.go.id/oauth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=authorization_code" \
  -d "code=AUTH_CODE" \
  -d "redirect_uri=REDIRECT_URI" \
  -d "client_id=CLIENT_ID" \
  -d "client_secret=CLIENT_SECRET" \
  -d "code_verifier=CODE_VERIFIER"
```

### Test UserInfo Endpoint
```bash
curl https://face.ahu.go.id/oauth/userinfo \
  -H "Authorization: Bearer ACCESS_TOKEN"
```

## Support

Untuk bantuan teknis:
- Email: support@example.com
- Documentation: https://face.ahu.go.id/api/docs/
