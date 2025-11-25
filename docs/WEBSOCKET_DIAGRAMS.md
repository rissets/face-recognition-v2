# WebSocket Architecture Diagram

## System Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                          WEBSOCKET ARCHITECTURE                             │
│                      Face Recognition Authentication                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


┌───────────────┐         ┌───────────────┐         ┌───────────────────────┐
│               │         │               │         │                       │
│    CLIENT     │         │   REST API    │         │  WEBSOCKET CONSUMER   │
│               │         │               │         │                       │
└───────┬───────┘         └───────┬───────┘         └───────────┬───────────┘
        │                         │                             │
        │                         │                             │
        │ 1. Create Session       │                             │
        │ POST /enrollment/       │                             │
        ├────────────────────────>│                             │
        │                         │                             │
        │ 2. Session Response     │                             │
        │ {session_token,         │                             │
        │  websocket_url}         │                             │
        │<────────────────────────┤                             │
        │                         │                             │
        │                         │                             │
        │ 3. WebSocket Connect                                  │
        │ ws://host/ws/auth/process-image/{token}/              │
        ├───────────────────────────────────────────────────────>│
        │                         │                             │
        │ 4. Connection Established                             │
        │ {type: "connection_established"}                      │
        │<───────────────────────────────────────────────────────┤
        │                         │                             │
        │                         │                             │
        │ 5. Send Frame 1         │                             │
        │ {type: "frame",         │                             │
        │  image: "base64..."}    │                             │
        ├───────────────────────────────────────────────────────>│
        │                         │                             │
        │                         │         Process Frame       │
        │                         │         - Face Detection    │
        │                         │         - Liveness Check    │
        │                         │         - Quality Check     │
        │                         │                             │
        │ 6. Frame Processed      │                             │
        │ {success, liveness,     │                             │
        │  quality, progress}     │                             │
        │<───────────────────────────────────────────────────────┤
        │                         │                             │
        │                         │                             │
        │ 7. Send Frame 2         │                             │
        ├───────────────────────────────────────────────────────>│
        │                         │                             │
        │ 8. Frame Processed      │                             │
        │<───────────────────────────────────────────────────────┤
        │                         │                             │
        │                         │                             │
        │ 9. Send Frame 3         │                             │
        ├───────────────────────────────────────────────────────>│
        │                         │                             │
        │                         │         Complete!           │
        │                         │         - Save to DB        │
        │                         │         - Add to Engine     │
        │                         │         - Encrypt Response  │
        │                         │                             │
        │ 10. Enrollment Complete │                             │
        │ {enrollment_id,         │                             │
        │  encrypted_data}        │                             │
        │<───────────────────────────────────────────────────────┤
        │                         │                             │
        │ 11. Decrypt Response    │                             │
        │ with secret_key         │                             │
        │ → {id, timestamp}       │                             │
        │                         │                             │
        │ 12. Close Connection    │                             │
        ├───────────────────────────────────────────────────────>│
        │                         │                             │
        ▼                         ▼                             ▼
```

## Encryption Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                           ENCRYPTION FLOW                                   │
│                         Double Encryption System                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


SERVER SIDE                                             CLIENT SIDE
─────────────                                           ───────────

1. Generate Payload
   ┌────────────────────┐
   │ {                  │
   │   id: "abc123",    │
   │   timestamp: "...", │
   │   session_type: "" │
   │ }                  │
   └────────┬───────────┘
            │
            │ Convert to JSON
            ▼
   ┌────────────────────┐
   │ JSON String        │
   └────────┬───────────┘
            │
            │ Encrypt with API Key
            │ Algorithm: AES-256-CBC
            │ Key: SHA256(api_key)
            │ IV: Random 16 bytes
            ▼
   ┌────────────────────┐
   │ IV + Ciphertext    │
   └────────┬───────────┘
            │
            │ Encode Base64
            ▼
   ┌────────────────────┐
   │ Base64 String      │────────────────────────────────>   Receive
   └────────────────────┘                                   Encrypted Data
                                                                    │
                                                                    │
                                                           Decode Base64
                                                                    │
                                                                    ▼
                                                          ┌────────────────────┐
                                                          │ IV + Ciphertext    │
                                                          └─────────┬──────────┘
                                                                    │
                                                                    │ Extract IV
                                                                    │ Extract Ciphertext
                                                                    ▼
                                                          ┌────────────────────┐
                                                          │ Decrypt with       │
                                                          │ Secret Key         │
                                                          │ Key: SHA256(secret)│
                                                          └─────────┬──────────┘
                                                                    │
                                                                    ▼
                                                          ┌────────────────────┐
                                                          │ Padded Data        │
                                                          └─────────┬──────────┘
                                                                    │
                                                                    │ Remove Padding
                                                                    ▼
                                                          ┌────────────────────┐
                                                          │ JSON String        │
                                                          └─────────┬──────────┘
                                                                    │
                                                                    │ Parse JSON
                                                                    ▼
                                                          ┌────────────────────┐
                                                          │ {                  │
                                                          │   id: "abc123",    │
                                                          │   timestamp: "...", │
                                                          │   session_type: "" │
                                                          │ }                  │
                                                          └────────────────────┘
                                                          
                                                                DECRYPTED!
```

## Component Interaction

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                        COMPONENT INTERACTION                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


┌───────────────────────────────────────────────────────────────────────────┐
│                                                                           │
│                              CLIENT LAYER                                 │
│                                                                           │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────────┐    │
│  │  Web Browser    │  │  Mobile App      │  │  Python Script      │    │
│  │  (JavaScript)   │  │  (React Native)  │  │  (test script)      │    │
│  └────────┬────────┘  └────────┬─────────┘  └──────────┬──────────┘    │
│           │                    │                        │                │
└───────────┼────────────────────┼────────────────────────┼────────────────┘
            │                    │                        │
            └────────────────────┴────────────────────────┘
                                 │
                      ┌──────────▼──────────┐
                      │    WebSocket        │
                      │   ws://host/ws/...  │
                      └──────────┬──────────┘
                                 │
┌────────────────────────────────┼────────────────────────────────────────────┐
│                                │                                            │
│                       APPLICATION LAYER                                     │
│                                │                                            │
│               ┌────────────────▼──────────────┐                            │
│               │  AuthProcessConsumer          │                            │
│               │  (WebSocket Consumer)         │                            │
│               │                                │                            │
│               │  - Accept Connection          │                            │
│               │  - Validate Session           │                            │
│               │  - Process Frames             │                            │
│               │  - Encrypt Response           │                            │
│               └────────┬───────────┬──────────┘                            │
│                        │           │                                       │
│                ┌───────▼───┐  ┌───▼────────┐                              │
│                │  Views    │  │  Models    │                              │
│                │  (REST)   │  │            │                              │
│                └───────────┘  └────────────┘                              │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
                                 │
┌────────────────────────────────┼────────────────────────────────────────────┐
│                                │                                            │
│                        PROCESSING LAYER                                     │
│                                │                                            │
│               ┌────────────────▼──────────────┐                            │
│               │  FaceRecognitionEngine        │                            │
│               │                                │                            │
│               │  - Face Detection             │                            │
│               │  - Liveness Check             │                            │
│               │  - Quality Assessment         │                            │
│               │  - Embedding Generation       │                            │
│               │  - Face Recognition           │                            │
│               └────────┬──────────────────────┘                            │
│                        │                                                   │
└────────────────────────┼───────────────────────────────────────────────────┘
                         │
┌────────────────────────┼───────────────────────────────────────────────────┐
│                        │                                                   │
│                   STORAGE LAYER                                            │
│                        │                                                   │
│     ┌──────────────────▼──────────┐     ┌────────────────────────┐       │
│     │      PostgreSQL              │     │      FAISS Index       │       │
│     │                              │     │                        │       │
│     │  - Sessions                  │     │  - Face Embeddings     │       │
│     │  - Enrollments               │     │  - Fast Search         │       │
│     │  - Users                     │     │                        │       │
│     │  - Logs                      │     │                        │       │
│     └──────────────────────────────┘     └────────────────────────┘       │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

## Session Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                          SESSION LIFECYCLE                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


    START
      │
      ▼
┌──────────────┐
│   Create     │  REST API: POST /api/auth/enrollment/
│   Session    │  or POST /api/auth/authentication/
└──────┬───────┘
       │
       │ Generate session_token
       │ Create database record
       │ Initialize metadata
       ▼
┌──────────────┐
│   Active     │  Status: "active"
│              │  Awaiting WebSocket connection
└──────┬───────┘
       │
       │ Client connects to WebSocket
       ▼
┌──────────────┐
│  Connected   │  Status: "processing"
│              │  Receiving frames
└──────┬───────┘
       │
       │ Processing frames
       │ ├─ Frame 1 → Quality check → Liveness check
       │ ├─ Frame 2 → Quality check → Liveness check
       │ └─ Frame 3 → Quality check → Liveness check
       │
       ▼
       ?
    Enough     ────NO───> Continue processing
    frames?              (Max 120 frames)
       │                       │
       │YES                    │
       ▼                       ▼
┌──────────────┐        ┌──────────────┐
│  Completed   │        │    Failed    │
│              │        │              │
│ Status:      │        │ Status:      │
│ "completed"  │        │ "failed"     │
│              │        │              │
│ - Save data  │        │ - Log error  │
│ - Encrypt    │        │ - Cleanup    │
│ - Return ID  │        │              │
└──────┬───────┘        └──────┬───────┘
       │                       │
       │ Close WebSocket       │
       ▼                       ▼
┌──────────────┐        ┌──────────────┐
│   Expired    │◄───────│  Cancelled   │
│              │  Time  │              │
│ After 5 min  │  out   │ User abort   │
└──────────────┘        └──────────────┘
       │
       │ Cleanup and archive
       ▼
      END


POSSIBLE STATUSES:
─────────────────
• active       → Session created, waiting for connection
• processing   → WebSocket connected, processing frames
• completed    → Successfully completed (enrollment/auth)
• failed       → Processing failed (no face, quality issues)
• expired      → Timed out (5 minutes)
• cancelled    → User cancelled
• disconnected → WebSocket disconnected unexpectedly
```

## Security Layers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                           SECURITY LAYERS                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


Layer 1: Transport Security
───────────────────────────
    ┌─────────────────────────┐
    │   WSS (WebSocket TLS)   │  ← Encrypted transport
    │   HTTPS (REST API)      │  ← SSL/TLS certificates
    └─────────────────────────┘

Layer 2: Authentication
───────────────────────
    ┌─────────────────────────┐
    │   API Key               │  ← Bearer token in header
    │   Session Token         │  ← Unique per session
    └─────────────────────────┘

Layer 3: Session Validation
────────────────────────────
    ┌─────────────────────────┐
    │   Status Check          │  ← Active/Processing only
    │   Expiration Check      │  ← Max 5 minutes
    │   Client Matching       │  ← Session belongs to client
    └─────────────────────────┘

Layer 4: Rate Limiting
──────────────────────
    ┌─────────────────────────┐
    │   Frame Rate            │  ← Max 10 FPS
    │   Frame Count           │  ← Max 120 per session
    │   Throttling            │  ← 100ms minimum interval
    └─────────────────────────┘

Layer 5: Data Encryption
────────────────────────
    ┌─────────────────────────┐
    │   AES-256-CBC           │  ← Symmetric encryption
    │   SHA-256 Key Derivation│  ← From API/Secret key
    │   Random IV             │  ← Per encryption
    └─────────────────────────┘

Layer 6: Input Validation
─────────────────────────
    ┌─────────────────────────┐
    │   Image Format          │  ← JPEG/PNG only
    │   Image Size            │  ← Max size check
    │   Face Detection        │  ← Must have face
    │   Quality Check         │  ← Minimum quality
    └─────────────────────────┘

Layer 7: Liveness Detection
────────────────────────────
    ┌─────────────────────────┐
    │   Blink Detection       │  ← Eye aspect ratio
    │   Motion Detection      │  ← Head movement
    │   Quality Variance      │  ← Natural variations
    └─────────────────────────┘
```

This architecture provides:
✅ Multiple layers of security
✅ Real-time processing
✅ Efficient communication
✅ Scalable design
✅ Clear separation of concerns
