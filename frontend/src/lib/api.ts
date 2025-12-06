import type {
  Job,
  JobCreateRequest,
  JobListResponse,
  JobLogsResponse,
  JobStatistics,
  QueueStatus,
  HealthStatus,
  TokenResponse,
  LoginRequest,
  RegisterRequest,
  SetupStatus,
  GlobalSettings,
  SettingsUpdateRequest,
  FileBrowserResponse,
  User,
} from '@/types';

const API_BASE = '/api';

/**
 * Get the stored access token
 */
function getAccessToken(): string | null {
  return localStorage.getItem('access_token');
}

/**
 * Get the stored refresh token
 */
function getRefreshToken(): string | null {
  return localStorage.getItem('refresh_token');
}

/**
 * Store tokens in localStorage
 */
function storeTokens(tokens: TokenResponse): void {
  localStorage.setItem('access_token', tokens.access_token);
  localStorage.setItem('refresh_token', tokens.refresh_token);
}

/**
 * Clear tokens from localStorage
 */
export function clearTokens(): void {
  localStorage.removeItem('access_token');
  localStorage.removeItem('refresh_token');
}

/**
 * Check if user is authenticated
 */
export function isAuthenticated(): boolean {
  return !!getAccessToken();
}

/**
 * Base fetch wrapper with auth and error handling
 */
async function apiFetch<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const token = getAccessToken();
  
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    ...options.headers,
  };

  if (token) {
    (headers as Record<string, string>)['Authorization'] = `Bearer ${token}`;
  }

  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers,
  });

  if (response.status === 401) {
    // Try to refresh token
    const refreshToken = getRefreshToken();
    if (refreshToken) {
      try {
        const refreshResponse = await fetch(`${API_BASE}/auth/refresh`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ refresh_token: refreshToken }),
        });
        
        if (refreshResponse.ok) {
          const tokens: TokenResponse = await refreshResponse.json();
          storeTokens(tokens);
          
          // Retry original request
          (headers as Record<string, string>)['Authorization'] = `Bearer ${tokens.access_token}`;
          const retryResponse = await fetch(`${API_BASE}${endpoint}`, {
            ...options,
            headers,
          });
          
          if (!retryResponse.ok) {
            throw new Error(`API Error: ${retryResponse.status}`);
          }
          return retryResponse.json();
        }
      } catch {
        clearTokens();
        window.location.href = '/login';
      }
    }
    clearTokens();
    window.location.href = '/login';
  }

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `API Error: ${response.status}`);
  }

  // Handle 204 No Content
  if (response.status === 204) {
    return null as T;
  }

  return response.json();
}

// =============================================================================
// Auth API
// =============================================================================

export const authApi = {
  checkSetup: () => apiFetch<SetupStatus>('/auth/setup'),

  register: async (data: RegisterRequest): Promise<TokenResponse> => {
    const tokens = await apiFetch<TokenResponse>('/auth/register', {
      method: 'POST',
      body: JSON.stringify(data),
    });
    storeTokens(tokens);
    return tokens;
  },

  login: async (data: LoginRequest): Promise<TokenResponse> => {
    const tokens = await apiFetch<TokenResponse>('/auth/login', {
      method: 'POST',
      body: JSON.stringify(data),
    });
    storeTokens(tokens);
    return tokens;
  },

  refresh: async (): Promise<TokenResponse> => {
    const refreshToken = getRefreshToken();
    if (!refreshToken) {
      throw new Error('No refresh token available');
    }
    const tokens = await apiFetch<TokenResponse>('/auth/refresh', {
      method: 'POST',
      body: JSON.stringify({ refresh_token: refreshToken }),
    });
    storeTokens(tokens);
    return tokens;
  },

  logout: () => {
    clearTokens();
    window.location.href = '/login';
  },

  getCurrentUser: () => apiFetch<User>('/auth/me'),
};

// =============================================================================
// Jobs API
// =============================================================================

export const jobsApi = {
  list: (params?: {
    status?: string;
    type?: string;
    page?: number;
    page_size?: number;
  }) => {
    const searchParams = new URLSearchParams();
    if (params?.status) searchParams.set('status', params.status);
    if (params?.type) searchParams.set('type', params.type);
    if (params?.page) searchParams.set('page', params.page.toString());
    if (params?.page_size) searchParams.set('page_size', params.page_size.toString());
    
    const query = searchParams.toString();
    return apiFetch<JobListResponse>(`/jobs${query ? `?${query}` : ''}`);
  },

  get: (id: string) => apiFetch<Job>(`/jobs/${id}`),

  create: (data: JobCreateRequest) =>
    apiFetch<Job>('/jobs', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  createBatch: (jobs: JobCreateRequest[]) =>
    apiFetch<Job[]>('/jobs/batch', {
      method: 'POST',
      body: JSON.stringify(jobs),
    }),

  cancel: (id: string) =>
    apiFetch<Job>(`/jobs/${id}/cancel`, { method: 'POST' }),

  retry: (id: string) =>
    apiFetch<Job>(`/jobs/${id}/retry`, { method: 'POST' }),

  delete: (id: string) =>
    apiFetch<void>(`/jobs/${id}`, { method: 'DELETE' }),

  getLogs: (id: string) => apiFetch<JobLogsResponse>(`/jobs/${id}/logs`),

  getOutput: (id: string) => apiFetch<{ content: string }>(`/jobs/${id}/output`),

  getStatistics: () => apiFetch<JobStatistics>('/stats'),
};

// =============================================================================
// System API
// =============================================================================

export const systemApi = {
  health: () => apiFetch<HealthStatus>('/health'),
  queueStatus: () => apiFetch<QueueStatus>('/queue/status'),
};

// =============================================================================
// Settings API
// =============================================================================

export const settingsApi = {
  get: () => apiFetch<GlobalSettings>('/settings'),

  update: (data: SettingsUpdateRequest) =>
    apiFetch<GlobalSettings>('/settings', {
      method: 'PUT',
      body: JSON.stringify(data),
    }),
};

// =============================================================================
// Files API
// =============================================================================

export const filesApi = {
  browse: (path?: string, showHidden = false) => {
    const params = new URLSearchParams();
    if (path) params.set('path', path);
    if (showHidden) params.set('show_hidden', 'true');
    const query = params.toString();
    return apiFetch<FileBrowserResponse>(`/files${query ? `?${query}` : ''}`);
  },
};
