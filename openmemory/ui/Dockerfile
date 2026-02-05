# syntax=docker.io/docker/dockerfile:1

# Base stage for common setup
FROM node:18-alpine AS base

# Install dependencies for pnpm
RUN apk add --no-cache libc6-compat curl && \
    corepack enable && \
    corepack prepare pnpm@latest --activate

WORKDIR /app

FROM base AS deps

COPY package.json pnpm-lock.yaml ./

RUN pnpm install --frozen-lockfile

FROM base AS builder
WORKDIR /app

COPY --from=deps /app/node_modules ./node_modules
COPY --from=deps /app/pnpm-lock.yaml ./pnpm-lock.yaml
COPY . .

RUN cp next.config.dev.mjs next.config.mjs
# --- 关键修改开始 ---
# 1. 声明接收构建参数
ARG NEXT_PUBLIC_API_URL

# 2. 将其设置为环境变量，Next.js 构建时会自动抓取
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}

# 3. 删掉下面这一行，或者确保它在 ENV 之前执行（建议直接删掉）
# RUN cp .env.example .env 
# --- 关键修改结束 ---

RUN pnpm build

FROM base AS runner
WORKDIR /app

ENV NODE_ENV=production

RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

COPY --chown=nextjs:nodejs entrypoint.sh /home/nextjs/entrypoint.sh
RUN chmod +x /home/nextjs/entrypoint.sh

USER nextjs

EXPOSE 3000
ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

# 注释掉或删掉这行，它就是 SyntaxError 的元凶
# ENTRYPOINT ["/home/nextjs/entrypoint.sh"] 

CMD ["node", "server.js"]
