# Cloud Run container for the voice-assistant Node server.
# Auth to Vertex AI is via the runtime service account (ADC) — no key file.
FROM node:22-alpine

WORKDIR /app

# Install production dependencies against the committed lockfile.
COPY package.json package-lock.json ./
RUN npm ci --omit=dev

# App source.
COPY server ./server
COPY public ./public

ENV NODE_ENV=production
# Cloud Run injects PORT (default 8080); server/index.js already reads it.
EXPOSE 8080

CMD ["node", "server/index.js"]
