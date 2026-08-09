# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1: Vite build
# ═══════════════════════════════════════════════════════════════════════════════
FROM node:20-alpine AS builder

WORKDIR /ui

# Cache node_modules layer
COPY package.json ./
RUN npm install --frozen-lockfile

# Build production bundle
COPY . .
RUN npm run build


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 2: Nginx serve
# ═══════════════════════════════════════════════════════════════════════════════
FROM nginx:1.27-alpine

LABEL org.opencontainers.image.title="mlops-ui"
LABEL org.opencontainers.image.description="Multi-Agent MLOps React UI"

# Remove default nginx config
RUN rm /etc/nginx/conf.d/default.conf

# Custom nginx config with SPA fallback + API proxy
COPY --from=builder /ui/dist /usr/share/nginx/html

RUN cat > /etc/nginx/conf.d/mlops.conf << 'EOF'
server {
    listen 80;
    server_name _;

    root /usr/share/nginx/html;
    index index.html;

    # SPA fallback — all unknown routes → index.html
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Proxy /api/* to the orchestrator backend
    location /api/ {
        proxy_pass http://orchestrator-agent.mlops.svc.cluster.local:8000;
        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
    }

    # Security headers
    add_header X-Frame-Options       "SAMEORIGIN"   always;
    add_header X-Content-Type-Options "nosniff"      always;
    add_header X-XSS-Protection      "1; mode=block" always;
    add_header Referrer-Policy       "strict-origin-when-cross-origin" always;

    # Cache static assets aggressively
    location ~* \.(js|css|png|svg|ico|woff2?)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    gzip on;
    gzip_types text/plain text/css application/javascript application/json;
    gzip_min_length 1024;
}
EOF

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget -qO- http://localhost/index.html || exit 1

CMD ["nginx", "-g", "daemon off;"]
