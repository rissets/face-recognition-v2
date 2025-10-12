# SSL Certificate Setup

This directory is for SSL certificates when running in production with HTTPS.

## For Let's Encrypt (Recommended)

1. Install Certbot:
```bash
sudo apt-get update
sudo apt-get install certbot python3-certbot-nginx
```

2. Obtain certificates:
```bash
sudo certbot --nginx -d your-domain.com
```

3. Certificates will be automatically placed in `/etc/letsencrypt/live/your-domain.com/`

4. Copy to this directory:
```bash
cp /etc/letsencrypt/live/your-domain.com/fullchain.pem ./cert.pem
cp /etc/letsencrypt/live/your-domain.com/privkey.pem ./key.pem
```

## For Self-Signed Certificates (Development)

```bash
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout key.pem \
  -out cert.pem \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
```

## Nginx Configuration

Update nginx/conf.d/default.conf to include:

```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    
    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    
    # Your server configuration...
}

server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}
```