#!/bin/bash
# Test OIDC Face Authentication Flow
# Usage: ./run_oidc_test.sh

# Configuration
BASE_URL="http://127.0.0.1:8003"
API_KEY="frapi_YY7OEJn1FCyDoGiGLwiueTw79hkQWduGNy2L-XbsCB4"
SECRET_KEY="_lwZfcqmdsi5PtRLjmOeTDgTxP5JaRAN3r4i6IpCSOC6ndL536sO9ZuFVjbgLshbmuNKmBButy_wZgdyXEw-DA"
OIDC_CLIENT_ID="oidc_vBYjlMiaEUgUdnObhaetc37L-HzsF-_H"
OIDC_CLIENT_SECRET="Jv8VYQNN5ODN5lZloiE2MhZ8TFQzBJplwOg2th_bPwdk9ogKApyYfU2PUK8Trudt"
USER_ID="653384"

echo "============================================================"
echo "🔐 OIDC Face Authentication Test"
echo "============================================================"
echo ""
echo "📍 Base URL: $BASE_URL"
echo "📍 OIDC Client: $OIDC_CLIENT_ID"
echo "📍 User ID: $USER_ID"
echo ""

# Step 1: Test OIDC Discovery
echo "📡 Step 1: Testing OIDC Discovery..."
DISCOVERY=$(curl -s "$BASE_URL/.well-known/openid-configuration")
if echo "$DISCOVERY" | grep -q "issuer"; then
    echo "✅ OIDC Discovery successful!"
    echo "   Issuer: $(echo $DISCOVERY | python3 -c "import sys,json; print(json.load(sys.stdin).get('issuer','N/A'))")"
else
    echo "❌ OIDC Discovery failed!"
    exit 1
fi
echo ""

# Step 2: Test JWKS
echo "📡 Step 2: Testing JWKS..."
JWKS=$(curl -s "$BASE_URL/.well-known/jwks.json")
if echo "$JWKS" | grep -q "keys"; then
    echo "✅ JWKS available!"
else
    echo "❌ JWKS failed!"
fi
echo ""

# Step 3: Run face authentication with WebSocket
echo "📹 Step 3: Running Face Authentication..."
echo "   This will open camera for face recognition"
echo ""

cd "$(dirname "$0")"
source env/bin/activate

python test_oidc_face_auth.py \
    "$API_KEY" \
    "$SECRET_KEY" \
    "$BASE_URL" \
    "$OIDC_CLIENT_ID" \
    "$OIDC_CLIENT_SECRET" \
    "$USER_ID"

echo ""
echo "============================================================"
echo "Test completed!"
echo "============================================================"
