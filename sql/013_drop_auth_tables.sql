-- SQL Migration: Drop Email Auth Dependent Tables
-- Date: 2026-04-05
-- Purpose: Remove tables that require Supabase Auth (email signup)
-- 
-- This removes:
-- 1. users_profile - References auth.users (email signup required)
-- 2. reminders - Tied to auth pattern
-- 3. insight_snapshots - Optional, can be omitted
--
-- Only RUN if you want to use the BRIDGE TABLE pattern (service-role)
-- WITHOUT Supabase Auth integration

-- Drop reminders first (references users_profile via foreign key chain)
DROP TABLE IF EXISTS public.reminders CASCADE;

-- Drop insight_snapshots
DROP TABLE IF EXISTS public.insight_snapshots CASCADE;

-- Drop users_profile (references auth.users)
DROP TABLE IF EXISTS public.users_profile CASCADE;

-- Verify deletion
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
  AND table_name IN ('users_profile', 'reminders', 'insight_snapshots');

-- Expected output: (no rows - tables successfully dropped)

-- ============================================================================
-- REMAINING TABLES (BRIDGE PATTERN - No Auth Required)
-- ============================================================================
-- 
-- These tables remain and are fully functional WITHOUT Supabase Auth:
-- 
-- 1. ai_conversation_threads (TEXT user_id - no auth needed)
-- 2. ai_conversation_messages (TEXT user_id - no auth needed)  
-- 3. ai_message_embeddings (TEXT user_id - no auth needed)
-- 4. ai_conversation_quality_metrics (TEXT user_id - no auth needed)
--
-- All use TEXT user_id instead of UUID references to auth.users
-- This allows testing without email signup/authentication
