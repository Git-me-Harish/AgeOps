# ui/Dockerfile — Next.js 14 standalone server, replacing the V1
# static-build+nginx image (see configs/kubernetes/ingress.yaml).
FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci

FROM node:20-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV PORT=3000

# node:20-alpine already ships a non-root `node` user (uid 1000)
COPY --from=builder /app/public ./public
COPY --from=builder --chown=node:node /app/.next/standalone ./
COPY --from=builder --chown=node:node /app/.next/static ./.next/static

USER node
EXPOSE 3000

HEALTHCHECK --interval=20s --timeout=4s --start-period=15s --retries=3 \
  CMD wget -qO- http://localhost:3000/api/healthz || exit 1

CMD ["node", "server.js"]
