export type NavRoute =
  | 'dashboard'
  | 'analyze'
  | 'results'
  | 'history'
  | 'ml'
  | 'reports';

export interface VideoMeta {
  name: string;
  shot_type: string;
  /** Model classification; null for clips analysed before it existed. */
  classification?: ShotClassification | null;
  rating: number | null;
  frames: number | null;
  processed_at: string;
  analysis_exists: boolean;
  analysis_url?: string | null;
  results_url?: string;
}

export interface DatasetStats {
  total_videos: number;
  n_classes: number;
  class_names: string[];
  distribution: Record<string, number>;
  n_features: number;
  dataset_exists: boolean;
  has_ml_reports: {
    comparison: boolean;
    confusion: boolean;
    importance: boolean;
    summary: boolean;
  };
  ml?: MLInsights;
}

export interface MLInsights {
  available: boolean;
  dataset_size: number | null;
  n_classes: number | null;
  best_model: string | null;
  evaluation: string;
  accuracy: string | null;
  accuracy_raw: number | null;
  comparison_table: {
    model: string;
    accuracy: number | null;
    f1_macro: number | null;
    precision_macro: number | null;
    recall_macro: number | null;
  }[];
}

export interface ShotClassification {
  video_name?: string;
  /** Final class. "unknown" whenever the model was not confident. */
  shot_type: string;
  /** The model's top class before the confidence gate. */
  predicted: string | null;
  confidence: number | null;
  threshold: number;
  is_unknown: boolean;
  reason: string;
  reason_label: string;
  model_name: string | null;
  n_classes: number;
  probabilities: Record<string, number>;
}

export interface ResultsPayload {
  video_name: string;
  shot_type: string;
  shot_type_label: string;
  classification: ShotClassification | null;
  found: boolean;
  frames: number | null;
  metrics: Record<string, number | null>;
  features: Record<string, number | null>;
  impact_angles: Record<string, number | null>;
  impact_angle_labels: Record<string, string>;
  impact_angle_columns: string[];
  rating: {
    rating: number | null;
    classification: string | null;
    explanation: string | null;
    confidence: number | null;
    factors: {
      code: string;
      label: string;
      score: number | null;
      weight: number;
      weighted: number | null;
      comment: string;
    }[];
    strengths: string[];
    weaknesses: string[];
    unavailable_metrics: string[];
  } | null;
  injury: {
    summary: {
      safety_score: number;
      risk_level: string;
      coverage: number;
      n_flags: number;
      top_concern: string;
      summary: string;
    } | null;
    flags: {
      metric: string;
      label: string;
      side: string;
      severity: string;
      frames: number;
      pct: number;
      peak: number;
      units: string;
      worst_phase: string;
      description: string;
      cue: string;
    }[];
    available: boolean;
  };
  phase_timeline: {
    segments: { phase: string; start: number; end: number | null }[];
    total_frames: number;
    impact_frame: number | null;
    fps: number | null;
  } | null;
  charts: Record<string, string>;
  plot: string | null;
  has_analysis: boolean;
  analysis_url: string | null;
  output_base: string;
  ml?: MLInsights;
}

export const SHOT_TYPES = [
  'cut',
  'defence',
  'drive',
  'flick',
  'pull shot',
  'unknown',
] as const;
export type ShotType = (typeof SHOT_TYPES)[number];

export const SHOT_TYPE_LABELS: Record<string, string> = {
  cut: 'Cut',
  defence: 'Defence',
  drive: 'Drive',
  flick: 'Flick',
  'pull shot': 'Pull Shot',
  unknown: 'Unknown',
};

/** Human-readable label for a shot_type value from the API. */
export function shotLabel(shotType: string | null | undefined): string {
  if (!shotType) return 'Unknown';
  return SHOT_TYPE_LABELS[shotType.toLowerCase()] ?? shotType;
}

export const PHASE_COLORS: Record<string, string> = {
  Stance: '#3b82f6',
  Backlift: '#f59e0b',
  Downswing: '#10b981',
  Impact: '#ef4444',
  'Follow-through': '#8b5cf6',
};