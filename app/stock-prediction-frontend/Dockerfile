# Stage 1: Build
FROM node:18-alpine AS builder

# Copy package.json and package-lock.json to the container
COPY ./package*.json ./

# Install dependencies with --legacy-peer-deps to avoid peer dependency issues
RUN npm install --legacy-peer-deps

# Copy the rest of the application code
COPY . .

# Build the Next.js app
RUN npm run build

# Stage 2: Serve the app
FROM node:18-alpine AS runner

# Copy the built files from the builder stage
COPY --from=builder /app/ .

# Expose the port that the app runs on
EXPOSE 3000

# Start the Next.js app
CMD ["npm", "run", "start"]