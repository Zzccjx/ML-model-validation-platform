export interface User {
  id: string;
  email: string;
  full_name: string;
  role: string;
  is_active: boolean;
  created_at: string;
}

export interface Project {
  id: string;
  name: string;
  description: string | null;
  owner_id: string;
  created_at: string;
  updated_at: string;
}

export interface MLModel {
  id: string;
  project_id: string;
  name: string;
  filename: string;
  file_size: number;
  model_type: string | null;
  framework: string | null;
  feature_count: number | null;
  class_count: number | null;
  classes_json: any;
  metadata_json: Record<string, any> | null;
  uploaded_at: string;
}

export interface Dataset {
  id: string;
  project_id: string;
  name: string;
  filename: string;
  file_size: number;
  row_count: number | null;
  column_count: number | null;
  target_column: string | null;
  feature_columns_json: string[] | null;
  sensitive_columns_json: string[] | null;
  metadata_json: Record<string, any> | null;
  uploaded_at: string;
}

export interface ValidationRun {
  id: string;
  run_number: string;
  project_id: string;
  model_id: string;
  dataset_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'partial';
  overall_score: number | null;
  category_scores_json: Record<string, number> | null;
  config_json: Record<string, any> | null;
  started_at: string | null;
  completed_at: string | null;
  duration_seconds: number | null;
  error_message: string | null;
  created_at: string;
}

export interface Finding {
  id: string;
  validation_run_id: string;
  severity: 'critical' | 'high' | 'medium' | 'low' | 'info';
  category: string;
  title: string;
  description: string;
  metric_name: string | null;
  metric_value: number | null;
  threshold: string | null;
  recommendation: string | null;
  validator_name: string;
  validator_version: string;
  created_at: string;
}

export interface CategoryScore {
  name: string;
  score: number;
  label: string;
}

export interface ValidatorReadiness {
  name: string;
  ready: boolean;
  status_label: string;
  reason: string | null;
}

export interface CompatibilityCheckResponse {
  compatible: boolean;
  errors: string[];
  warnings: string[];
  model_info: Record<string, any>;
  dataset_info: Record<string, any>;
  validator_readiness: ValidatorReadiness[];
}

export interface DatasetPreview {
  total_rows: number;
  total_columns: number;
  columns: string[];
  preview_rows: Record<string, any>[];
  dtypes: Record<string, string>;
  missing_counts: Record<string, number>;
  duplicate_rows: number;
  numeric_columns: string[];
  categorical_columns: string[];
}

export interface AuthTokens {
  access_token: string;
  token_type: string;
}


