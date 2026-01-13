-- Supabase (Postgres) schema for AI Crop Assistant
-- Run this in the Supabase SQL editor or psql

-- Enable pgcrypto for gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Users table (optional if you want to sync with local users)
CREATE TABLE IF NOT EXISTS users (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name text,
  email text UNIQUE NOT NULL,
  created_at timestamptz DEFAULT now()
);

-- Predictions table to store disease detection or crop recommendations
CREATE TABLE IF NOT EXISTS predictions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_email text,
  type text,
  label text,
  confidence double precision,
  image_path text,
  payload jsonb,
  created_at timestamptz DEFAULT now()
);

-- Recommendations table (store recommended crops and metadata)
CREATE TABLE IF NOT EXISTS recommendations (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  crop text,
  score double precision,
  details jsonb,
  created_at timestamptz DEFAULT now()
);

-- Optional indexes
CREATE INDEX IF NOT EXISTS idx_predictions_user_email ON predictions (user_email);
CREATE INDEX IF NOT EXISTS idx_recommendations_crop ON recommendations (crop);
