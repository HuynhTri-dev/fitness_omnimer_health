# Stage 1: Build React app
FROM node:18-alpine AS builder

# Thư mục làm việc trong container
WORKDIR /app

# Copy package.json & package-lock.json trước để tận dụng cache
COPY package*.json ./

# Cài dependencies
RUN npm install

# Copy toàn bộ source code
COPY . .

# Build React app production
RUN npm run build

# Stage 2: Serve với NGINX
FROM nginx:alpine

# Copy build từ stage trước sang thư mục html của NGINX
COPY --from=builder /app/build /usr/share/nginx/html

# Copy config NGINX nếu muốn custom, ví dụ: /nginx.conf
# COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80
EXPOSE 80

# Command chạy NGINX
CMD ["nginx", "-g", "daemon off;"]
