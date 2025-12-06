// API Types matching the backend models

export type JobStatus = 'pending' | 'queued' | 'running' | 'done' | 'failed' | 'cancelled';
export type JobType = 'transcribe' | 'translate' | 'sync' | 'compare';
export type JobStage = 
  | 'queued' 
  | 'extracting_audio' 
  | 'transcribing' 
  | 'translating' 
  | 'syncing' 
  | 'comparing' 
  | 'finalizing' 
  | 'completed' 
  | 'failed';

export interface Job {
  id: string;
  type: JobType;
  status: JobStatus;
  stage: JobStage;
  progress: number;
  input_path: string;
  output_path: string | null;
  reference_path: string | null;
  source_language: string | null;
  target_language: string | null;
  model_size: string | null;
  compute_type: string | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
  started_at: string | null;
  completed_at: string | null;
}

export interface JobEvent {
  id: string;
  job_id: string;
  event_type: string;
  message: string | null;
  old_status: string | null;
  new_status: string | null;
  old_stage: string | null;
  new_stage: string | null;
  timestamp: string;
}

export interface JobCreateRequest {
  type: JobType;
  input_path: string;
  output_path?: string | null;
  reference_path?: string | null;
  source_language?: string | null;
  target_language?: string | null;
  model_size?: string;
  compute_type?: string;
  priority?: string;
  options?: Record<string, unknown>;
}

export interface JobListResponse {
  jobs: Job[];
  total: number;
  page: number;
  page_size: number;
  has_more: boolean;
}

export interface JobLogsResponse {
  job_id: string;
  logs: string | null;
  status: string;
  stage: string;
}

export interface JobStatistics {
  total_jobs: number;
  pending: number;
  running: number;
  completed: number;
  failed: number;
  cancelled: number;
}

export interface QueueStatus {
  queues: Record<string, unknown>;
  workers: unknown[];
  total_jobs: number;
  failed_jobs: number;
}

export interface HealthStatus {
  status: string;
  database: string;
  redis: string;
  version: string;
}

// Auth Types
export interface User {
  id: string;
  username: string;
  is_admin: boolean;
  is_active: boolean;
  created_at: string;
  last_login: string | null;
}

export interface LoginRequest {
  username: string;
  password: string;
}

export interface RegisterRequest {
  username: string;
  password: string;
}

export interface TokenResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

export interface SetupStatus {
  setup_required: boolean;
  message: string;
}

// Settings Types
export interface GlobalSettings {
  media_folders: string[];
  default_model: string;
  default_device: string;
  default_compute_type: string;
  default_language: string;
  default_translation_backend: string | null;
  worker_count: number;
  log_retention_days: number;
  job_retention_days: number;
  prefer_gpu: boolean;
  updated_at: string;
}

export interface SettingsUpdateRequest {
  media_folders?: string[];
  default_model?: string;
  default_device?: string;
  default_compute_type?: string;
  default_language?: string;
  default_translation_backend?: string | null;
  worker_count?: number;
  log_retention_days?: number;
  job_retention_days?: number;
  prefer_gpu?: boolean;
}

// File Browser Types
export interface FileItem {
  name: string;
  path: string;
  is_directory: boolean;
  size: number | null;
  modified: string | null;
  extension: string | null;
}

export interface FileBrowserResponse {
  current_path: string;
  parent_path: string | null;
  items: FileItem[];
  total_items: number;
}
