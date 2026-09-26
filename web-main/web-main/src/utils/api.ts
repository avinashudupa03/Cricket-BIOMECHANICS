import type { DatasetStats, MLInsights, ResultsPayload, VideoMeta } from '../types';

export type { DatasetStats, MLInsights, ResultsPayload, VideoMeta };

export interface ReportsPayload {
  eda_images: { name: string; url: string }[];
  images: Record<string, string | null>;
  documents: Record<string, string>;
  stats: DatasetStats;
}

export interface ProgressPayload {
  state: string;
  ok: boolean | null;
  steps: string[];
  current: string;
  error: string | null;
  video_name: string | null;
  shot_type: string;
}

export interface UploadResponse {
  status: string;
  job_id?: string;
  video_name?: string;
  shot_type?: string;
  message?: string;
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    throw new Error(`${url} -> HTTP ${res.status}`);
  }
  return (await res.json()) as T;
}

export const api = {
  stats: () => request<DatasetStats>('/api/stats'),
  videos: () =>
    request<{ count: number; videos: VideoMeta[] }>('/api/videos'),
  history: () =>
    request<{ total: number; rated_count: number; videos: VideoMeta[] }>(
      '/api/history'
    ),
  ml: () => request<MLInsights>('/api/ml'),
  reports: () => request<ReportsPayload>('/api/reports'),
  results: (videoName: string) =>
    request<ResultsPayload>(`/api/results/${encodeURIComponent(videoName)}`),
  progress: (jobId: string) =>
    request<ProgressPayload>(`/api/progress/${encodeURIComponent(jobId)}`),
  upload: (file: File, shotType: string) => {
    const form = new FormData();
    form.append('video', file);
    form.append('shot_type', shotType);
    return request<UploadResponse>('/api/upload', { method: 'POST', body: form });
  },
};

export function fmt(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return Number(value).toFixed(digits);
}