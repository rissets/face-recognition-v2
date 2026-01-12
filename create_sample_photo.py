#!/usr/bin/env python3
"""
Helper script to create a sample old profile photo and register user
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sys
import requests
import json
from pathlib import Path

def create_sample_photo(output_path: str, text: str = "OLD PHOTO", size=(640, 480)):
    """Create a sample photo with text overlay"""
    # Create a colored background
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    
    # Create gradient background (blue to cyan)
    for y in range(size[1]):
        color_value = int(100 + (y / size[1]) * 155)
        img[y, :] = [color_value, color_value // 2, 50]
    
    # Convert to PIL for text drawing
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    
    # Try to use a font, fallback to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 60)
    except:
        font = ImageFont.load_default()
    
    # Draw text
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    position = ((size[0] - text_width) // 2, (size[1] - text_height) // 2)
    
    # Draw white text with black outline
    for offset in [(-2,-2), (-2,2), (2,-2), (2,2)]:
        draw.text((position[0]+offset[0], position[1]+offset[1]), text, font=font, fill=(0,0,0))
    draw.text(position, text, font=font, fill=(255,255,255))
    
    # Add timestamp
    timestamp_text = "Sample Photo 2024"
    try:
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except:
        small_font = font
    
    draw.text((20, size[1] - 40), timestamp_text, font=small_font, fill=(255,255,255))
    
    # Convert back to OpenCV format
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    # Draw a simple face-like shape in the center
    center_x, center_y = size[0] // 2, size[1] // 2
    
    # Face oval
    cv2.ellipse(img, (center_x, center_y), (120, 160), 0, 0, 360, (255, 220, 180), -1)
    cv2.ellipse(img, (center_x, center_y), (120, 160), 0, 0, 360, (200, 180, 150), 3)
    
    # Eyes
    cv2.circle(img, (center_x - 40, center_y - 30), 15, (50, 50, 50), -1)
    cv2.circle(img, (center_x + 40, center_y - 30), 15, (50, 50, 50), -1)
    cv2.circle(img, (center_x - 40, center_y - 30), 8, (255, 255, 255), -1)
    cv2.circle(img, (center_x + 40, center_y - 30), 8, (255, 255, 255), -1)
    
    # Nose
    cv2.line(img, (center_x, center_y - 10), (center_x - 10, center_y + 20), (200, 180, 150), 2)
    
    # Mouth
    cv2.ellipse(img, (center_x, center_y + 40), (40, 20), 0, 0, 180, (200, 100, 100), 2)
    
    # Save image
    cv2.imwrite(output_path, img)
    print(f"✅ Sample photo created: {output_path}")
    print(f"   Size: {size[0]}x{size[1]}")
    return output_path


def authenticate_client(base_url: str, api_key: str, secret_key: str) -> str:
    """Authenticate client and get JWT token"""
    url = f"{base_url}/api/core/auth/client/"
    data = {"api_key": api_key, "api_secret": secret_key}
    
    print(f"🔑 Authenticating client...")
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        
        auth_data = response.json()
        jwt_token = auth_data.get("access_token")
        print(f"✅ Client authenticated successfully!")
        print(f"   Token: {jwt_token[:20]}..." if jwt_token else "   Token: None")
        return jwt_token
    except requests.exceptions.HTTPError as e:
        print(f"❌ Authentication failed!")
        print(f"   Status Code: {e.response.status_code}")
        print(f"   Response: {e.response.text}")
        return None
    except Exception as e:
        print(f"❌ Authentication error: {e}")
        return None


def create_user_with_old_photo(base_url: str, user_id: str, old_photo_path: str,
                                api_key: str = None, secret_key: str = None,
                                name: str = None, email: str = None, phone: str = None):
    """Create user via API with old profile photo"""
    
    # Step 1: Authenticate to get JWT token
    if not api_key or not secret_key:
        print("❌ Error: API key and secret key required")
        return None
    
    jwt_token = authenticate_client(base_url, api_key, secret_key)
    if not jwt_token:
        return None
    
    print()
    
    # Step 2: Create user
    if not name:
        name = f"Test User {user_id}"
    if not email:
        email = f"{user_id}@example.com"
    if not phone:
        phone = f"+62812345{user_id[-5:]}"
    
    # Prepare multipart form data
    url = f"{base_url}/api/clients/users/"
    
    files = {
        'old_profile_photo': open(old_photo_path, 'rb')
    }
    
    data = {
        'external_user_id': user_id,
        'face_auth_enabled': 'true',
    }
    
    # Add profile data as JSON string
    profile_data = {
        'name': name,
        'email': email,
        'phone': phone,
    }
    data['profile'] = json.dumps(profile_data)
    
    print(f"📡 Creating user via API...")
    print(f"   URL: {url}")
    print(f"   User ID: {user_id}")
    print(f"   Name: {name}")
    print(f"   Email: {email}")
    print(f"   Phone: {phone}")
    print(f"   Old Photo: {old_photo_path}")
    
    # Prepare headers with JWT token
    headers = {
        'Authorization': f'JWT {jwt_token}'
    }
    
    try:
        response = requests.post(url, files=files, data=data, headers=headers)
        
        response.raise_for_status()
        user_data = response.json()
        
        print(f"\n✅ User created successfully!")
        print(f"   ID: {user_data.get('id')}")
        print(f"   External User ID: {user_data.get('external_user_id')}")
        if user_data.get('old_profile_photo_url'):
            print(f"   Old Photo URL: {user_data.get('old_profile_photo_url')}")
        print(f"   Face Auth Enabled: {user_data.get('face_auth_enabled', False)}")
        return user_data
            
    except requests.exceptions.HTTPError as e:
        print(f"\n❌ Failed to create user!")
        print(f"   Status Code: {e.response.status_code}")
        print(f"   Response: {e.response.text}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request failed: {e}")
        return None
    finally:
        files['old_profile_photo'].close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Create sample old photo and optionally register user',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Just create photo
  python create_sample_photo.py old_photo.jpg
  
  # Use existing photo to create user (with authentication)
  python create_sample_photo.py --existing-photo my_photo.jpg --user-id user123 --api-key YOUR_KEY --secret-key YOUR_SECRET --base-url http://localhost:8000
  
  # Create photo and register user
  python create_sample_photo.py old_photo.jpg --user-id user123 --api-key YOUR_KEY --secret-key YOUR_SECRET --base-url http://localhost:8000
  
  # With custom user details
  python create_sample_photo.py photo.jpg --user-id john --name "John Doe" --email john@example.com --api-key YOUR_KEY --secret-key YOUR_SECRET --base-url http://localhost:8000
        """
    )
    
    parser.add_argument('output', nargs='?', default='old_photo_sample.jpg',
                       help='Output photo filename (default: old_photo_sample.jpg)')
    parser.add_argument('--text', default='OLD PHOTO',
                       help='Text to display on photo (default: OLD PHOTO)')
    parser.add_argument('--existing-photo', 
                       help='Use existing photo file instead of creating new one')
    parser.add_argument('--user-id', help='User ID to create')
    parser.add_argument('--api-key', help='API key for authentication')
    parser.add_argument('--secret-key', help='Secret key for encryption')
    parser.add_argument('--name', help='User name')
    parser.add_argument('--email', help='User email')
    parser.add_argument('--phone', help='User phone')
    parser.add_argument('--base-url', default='http://localhost:8000',
                       help='Base URL of API (default: http://localhost:8000)')
    
    args = parser.parse_args()
    
    # Determine which photo to use
    if args.existing_photo:
        # Use existing photo
        photo_path = args.existing_photo
        if not Path(photo_path).exists():
            print(f"❌ Error: Photo file not found: {photo_path}")
            sys.exit(1)
        
        # Validate it's a valid image
        img = cv2.imread(photo_path)
        if img is None:
            print(f"❌ Error: Invalid image file: {photo_path}")
            sys.exit(1)
        
        print(f"✅ Using existing photo: {photo_path}")
        print(f"   Size: {img.shape[1]}x{img.shape[0]}")
        file_size = Path(photo_path).stat().st_size
        print(f"   File size: {file_size / 1024:.1f} KB")
    else:
        # Create sample photo
        photo_path = create_sample_photo(args.output, args.text)
    
    # If user_id provided, create user via API
    if args.user_id:
        if not args.api_key or not args.secret_key:
            print("\n❌ Error: --api-key and --secret-key are required when creating user")
            print("   Usage: --user-id USER_ID --api-key YOUR_KEY --secret-key YOUR_SECRET")
            sys.exit(1)
        
        print("\n" + "="*60)
        create_user_with_old_photo(
            base_url=args.base_url,
            user_id=args.user_id,
            old_photo_path=photo_path,
            api_key=args.api_key,
            secret_key=args.secret_key,
            name=args.name,
            email=args.email,
            phone=args.phone
        )
        print("="*60)
        print(f"\n✅ User ready for enrollment!")
        print(f"\n🎥 Run enrollment test:")
        print(f"   python test_websocket_auth.py {args.api_key} {args.secret_key} {args.base_url} enrollment {args.user_id}")
    else:
        print(f"\nℹ️  Photo ready but user not registered.")
        print(f"   To create user, add: --user-id <user_id> --base-url http://localhost:8000")
        print(f"\n   Or use manually:")
        print(f"   python test_websocket_auth.py <API_KEY> <SECRET_KEY> <BASE_URL> enrollment <user_id> {photo_path}")
