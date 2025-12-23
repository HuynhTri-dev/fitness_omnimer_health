# Redis Caching Implementation Plan

## Overview

This document outlines the strategy for implementing Redis caching across various domain routes in the OmniMer Health Server. The primary goal is to cache `GET` requests for static or semi-static data (Master Data) to reduce database load and improve response times.

## General Strategy

- **Cache-Aside Pattern**: Attempt to read from cache first. If missing, read from DB, store in cache, and return.
- **Invalidation**: On any data mutation (`POST`, `PUT`, `DELETE`), invalidate relevant cache keys.
- **TTL (Time To Live)**: Set reasonable expiration times to ensure eventual consistency even if invalidation fails.

---

## Detailed Caching Map

### 1. Equipment (`src/domain/routes/equipment.route.ts`)

**Data Characteristics**: Static Master Data. Rarely changes.

- **Endpoints to Cache**:
  - `GET /` -> Key: `equipment:list` | TTL: 24h
- **Invalidation Triggers** (`POST`, `PUT`, `DELETE`):
  - Invalidate: `equipment:list`

### 2. Exercise Category (`src/domain/routes/exercise-category.route.ts`)

**Data Characteristics**: Static Master Data. Rarely changes.

- **Endpoints to Cache**:
  - `GET /` -> Key: `exercise_category:list` | TTL: 24h
  - `GET /:id` -> Key: `exercise_category:{id}` | TTL: 24h
- **Invalidation Triggers**:
  - `POST /` -> Invalidate `exercise_category:list`
  - `PUT /:id` -> Invalidate `exercise_category:{id}`, `exercise_category:list`
  - `DELETE /:id` -> Invalidate `exercise_category:{id}`, `exercise_category:list`

### 3. Exercise Type (`src/domain/routes/exercise-type.route.ts`)

**Data Characteristics**: Static Master Data. Rarely changes.

- **Endpoints to Cache**:
  - `GET /` -> Key: `exercise_type:list` | TTL: 24h
  - `GET /:id` -> Key: `exercise_type:{id}` | TTL: 24h
- **Invalidation Triggers**:
  - `POST /` -> Invalidate `exercise_type:list`
  - `PUT /:id` -> Invalidate `exercise_type:{id}`, `exercise_type:list`
  - `DELETE /:id` -> Invalidate `exercise_type:{id}`, `exercise_type:list`

### 4. Goal (`src/domain/routes/goal.route.ts`)

**Data Characteristics**: Mixed - System goals (static) & User goals (dynamic).

- **Endpoints to Cache**:
  - `GET /` -> Key: `goal:list` | TTL: 24h
  - `GET /:id` -> Key: `goal:{id}` | TTL: 24h
  - `GET /user/:userId` -> Key: `goal:user:{userId}` | TTL: 1h
- **Invalidation Triggers**:
  - `POST /` -> Invalidate `goal:list` (if system goal) or `goal:user:{userId}` (if user goal)
  - `PUT /:id` -> Invalidate `goal:{id}`, `goal:list`, check if related to user to invalidate `goal:user:{userId}`
  - `DELETE /:id` -> Invalidate `goal:{id}`, `goal:list`

### 5. Muscle (`src/domain/routes/muscle.route.ts`)

**Data Characteristics**: Static Anatomical Data. Very rarely changes.

- **Endpoints to Cache**:
  - `GET /` -> Key: `muscle:list` | TTL: 7 days
  - `GET /:id` -> Key: `muscle:{id}` | TTL: 7 days
  - `GET /name` -> Key: `muscle:name:{query_name}` | TTL: 7 days
- **Invalidation Triggers**:
  - `POST /` -> Invalidate `muscle:list`
  - `PUT /:id` -> Invalidate `muscle:{id}`, `muscle:list`, `muscle:name:*` (scan delete recommended or short TTL on name search)
  - `DELETE /:id` -> Invalidate `muscle:{id}`, `muscle:list`

### 6. Exercise (`src/domain/routes/exercise.route.ts`)

**Data Characteristics**: Core Content. Frequent reads. Potentially large dataset. Requests often contain filters/pagination.

- **Endpoints to Cache**:
  - `GET /` -> Key: `exercise:list:{hash_of_query_params}` | TTL: 1h
    - _Note_: If query params vary wildly, caching might result in low hit-rate/high memory. Consider caching only "default/empty" query or common filters.
  - `GET /:id` -> Key: `exercise:{id}` | TTL: 24h
- **Invalidation Triggers**:
  - `POST /` -> Invalidate `exercise:list*` (Wildcard delete)
  - `PUT /:id` -> Invalidate `exercise:{id}`, `exercise:list*`
  - `DELETE /:id` -> Invalidate `exercise:{id}`, `exercise:list*`

---

## Action Items

1.  **Redis Service Setup**: Ensure `RedisService` (or similar) handles connection and basic `get/set/del` operations.
2.  **Middleware Creation**: Create a `cache.middleware.ts` to wrap `GET` routes easily.
    ```typescript
    // Example signature
    export const cacheMiddleware = (keyGenerator: (req) => string, ttl: number) => ...
    ```
3.  **Controller Updates**: Update `create/update/delete` methods in controllers (or services) to call Redis invalidate logic upon success.
