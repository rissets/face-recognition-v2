# Face Recognition API Integration Guide

This document provides a comprehensive guide for integrating with the Face Recognition API. It covers the authentication process, core workflows like enrollment and authentication, and provides detailed information about each endpoint.

## 1. Overview

The API is organized into several functional groups, making it easy to navigate and understand. The primary categories are:

-   **Authentication**: Handles client and user authentication, including JWT management.
-   **Client Management**: Allows you to manage your client account and the users associated with it.
-   **Face Enrollment**: A session-based workflow for registering a user's face.
-   **Face Authentication**: A session-based workflow for verifying or identifying a user.
-   **System**: Provides endpoints for checking system health, status, and configuration.
-   **Analytics**: Exposes endpoints for retrieving usage metrics, audit logs, and performance data.
-   **Webhooks**: Manages webhook configurations, deliveries, and logs.
-   **Legacy Recognition**: Contains deprecated endpoints that should not be used in new integrations.

## 2. Authentication

API access is controlled via a two-tiered authentication system designed for security and flexibility.

1.  **Client Authentication (JWT Issuance)**: Your backend service must first authenticate itself to obtain a JSON Web Token (JWT). This is done by sending your **API Key** and **API Secret** to the `/api/core/auth/client/` endpoint. This token is short-lived for security.
2.  **JWT Authorization**: The obtained JWT must be included in the `Authorization` header for all subsequent API requests as a Bearer token (e.g., `Authorization: Bearer <your_jwt>`).

### Client Authentication Workflow

**Request:**

-   **Endpoint**: `POST /api/core/auth/client/`
-   **Body**:
    ```json
    {
      "api_key": "your_api_key",
      "api_secret": "your_api_secret"
    }
    ```

**Response (Success):**

A successful authentication returns an `access` and `refresh` token.

```json
{
  "access": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

## 3. Core Workflows

The primary functions of the API—enrollment and authentication—are session-based. You create a session, upload image frames to it, and receive feedback until the process is complete.

### 3.1. Face Enrollment

Enrollment is the process of registering a user's face with the system. It requires capturing a few high-quality images to create a reliable biometric template.

**Step 1: Create an Enrollment Session**

Initiate an enrollment session for a specific user.

-   **Endpoint**: `POST /api/auth/enrollment/`
-   **Body**:
    ```json
    {
      "user_id": "unique_user_identifier_from_your_system",
      "session_type": "webcam", // or "mobile", "api"
      "metadata": {
        "device_info": "Chrome on macOS"
      }
    }
    ```

**Response:**

The API returns a `session_token` that you will use for all subsequent requests in this session.

```json
{
  "session_token": "a-unique-session-token",
  "enrollment_id": "enrollment-guid",
  "status": "pending",
  "target_samples": 3,
  "expires_at": "2025-10-12T12:30:00Z",
  "message": "Enrollment session created. Stream frames to continue."
}
```

**Step 2: Process Face Image Frames**

Upload images (frames) to the session using the `session_token`. The API will process each frame for face detection, quality, and liveness.

-   **Endpoint**: `POST /api/auth/process-image/`
-   **Body**: `multipart/form-data`
    -   `session_token`: The token from Step 1.
    -   `image`: The image file or a base64-encoded data URL.

**Response (Frame Processed):**

The response provides immediate feedback on the frame and the overall progress of the enrollment.

```json
{
    "success": true,
    "session_status": "processing",
    "enrollment_progress": 33.3,
    "requires_more_frames": true,
    "frame_accepted": true,
    "quality_score": 0.95,
    "message": "Frame processed. Continue streaming..."
}
```

**Response (Enrollment Complete):**

Once enough valid frames have been collected, the enrollment is finalized.

```json
{
    "success": true,
    "session_status": "completed",
    "enrollment_complete": true,
    "message": "Enrollment completed successfully."
}
```

### 3.2. Face Authentication

Authentication can be performed for **verification** (a 1:1 match against a known user) or **identification** (a 1:N search against all enrolled users).

**Step 1: Create an Authentication Session**

-   **Endpoint**: `POST /api/auth/authentication/`
-   **Body (Verification)**: Provide the `user_id` you want to verify.
    ```json
    {
      "user_id": "unique_user_identifier_to_verify",
      "session_type": "webcam"
    }
    ```
-   **Body (Identification)**: Omit the `user_id` to search against all users.
    ```json
    {
      "session_type": "webcam"
    }
    ```

**Response:**

```json
{
  "session_token": "a-unique-auth-session-token",
  "status": "active",
  "session_type": "verification", // or "identification"
  "expires_at": "2025-10-12T12:40:00Z",
  "message": "Authentication session created. Stream frames to continue."
}
```

**Step 2: Process Face Image Frames**

This step is identical to the enrollment frame processing, using the authentication `session_token`.

-   **Endpoint**: `POST /api/auth/process-image/`

**Response (Authentication Success):**

If a match is found and liveness is confirmed, the API returns a success response with the matched user's details.

```json
{
    "success": true,
    "session_status": "completed",
    "result": "success",
    "matched_user": {
        "user_id": "unique_user_identifier"
    },
    "similarity_score": 0.95,
    "liveness_verified": true,
    "message": "Authentication successful."
}
```

**Response (Authentication Failed):**

If no match is found or liveness checks fail, the API returns a failure response.

```json
{
    "success": false,
    "session_status": "completed",
    "result": "failure",
    "error": "User not recognized or liveness check failed.",
    "message": "Authentication failed."
}
```

## 4. API Endpoint Reference

For a complete, interactive, and up-to-date list of all endpoints, their parameters, and response schemas, please refer to the generated **Swagger/OpenAPI documentation**. This is available at the `/api/schema/swagger-ui/` endpoint of the running application.

The documentation is organized by the following tags to help you find what you need quickly:

-   `Authentication`
-   `Client Management`
-   `Face Enrollment`
-   `Face Authentication`
-   `System`
-   `Analytics`
-   `Webhooks`
-   `Legacy Recognition` (Deprecated)

This interactive documentation should always be considered the single source of truth for the API.
