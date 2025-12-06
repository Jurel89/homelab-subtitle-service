import { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Save, RefreshCw, AlertCircle, CheckCircle, Plus, X } from 'lucide-react';
import { settingsApi } from '@/lib/api';
import type { SettingsUpdateRequest } from '@/types';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/Card';

export default function SettingsPage() {
  const queryClient = useQueryClient();
  const [formData, setFormData] = useState<SettingsUpdateRequest>({});
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [newFolder, setNewFolder] = useState('');

  const { data: settings, isLoading, error } = useQuery({
    queryKey: ['settings'],
    queryFn: settingsApi.get,
  });

  const updateMutation = useMutation({
    mutationFn: (data: SettingsUpdateRequest) => settingsApi.update(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['settings'] });
      setSaveSuccess(true);
      setTimeout(() => setSaveSuccess(false), 3000);
    },
  });

  useEffect(() => {
    if (settings) {
      setFormData({
        media_folders: settings.media_folders,
        default_model: settings.default_model,
        default_device: settings.default_device,
        default_compute_type: settings.default_compute_type,
        default_language: settings.default_language,
        default_translation_backend: settings.default_translation_backend,
        worker_count: settings.worker_count,
        log_retention_days: settings.log_retention_days,
        job_retention_days: settings.job_retention_days,
        prefer_gpu: settings.prefer_gpu,
      });
    }
  }, [settings]);

  const handleChange = <K extends keyof SettingsUpdateRequest>(
    field: K,
    value: SettingsUpdateRequest[K]
  ) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  const handleAddFolder = () => {
    if (newFolder.trim()) {
      const folders = [...(formData.media_folders || []), newFolder.trim()];
      handleChange('media_folders', folders);
      setNewFolder('');
    }
  };

  const handleRemoveFolder = (index: number) => {
    const folders = (formData.media_folders || []).filter((_, i) => i !== index);
    handleChange('media_folders', folders);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    updateMutation.mutate(formData);
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="h-8 w-8 animate-spin text-accent" />
      </div>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent className="pt-6">
          <div className="flex items-center gap-2 text-destructive">
            <AlertCircle className="h-5 w-5" />
            <span>Failed to load settings</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Settings</h1>
        <p className="text-muted-foreground mt-1">Configure global pipeline defaults</p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Media Folders */}
        <Card>
          <CardHeader>
            <CardTitle>Media Folders</CardTitle>
            <CardDescription>Directories to scan for media files</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              {(formData.media_folders || []).map((folder, index) => (
                <div key={index} className="flex items-center gap-2">
                  <Input value={folder} readOnly className="flex-1" />
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={() => handleRemoveFolder(index)}
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              ))}
            </div>
            <div className="flex items-center gap-2">
              <Input
                value={newFolder}
                onChange={(e) => setNewFolder(e.target.value)}
                placeholder="/path/to/media/folder"
                className="flex-1"
              />
              <Button type="button" variant="outline" size="sm" onClick={handleAddFolder}>
                <Plus className="h-4 w-4 mr-1" />
                Add
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Model Settings */}
        <Card>
          <CardHeader>
            <CardTitle>Whisper Model</CardTitle>
            <CardDescription>Configure transcription model defaults</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-1">Model Size</label>
                <select
                  value={formData.default_model || 'base'}
                  onChange={(e) => handleChange('default_model', e.target.value)}
                  className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                >
                  <option value="tiny">Tiny (fastest)</option>
                  <option value="base">Base</option>
                  <option value="small">Small</option>
                  <option value="medium">Medium</option>
                  <option value="large">Large</option>
                  <option value="large-v2">Large V2</option>
                  <option value="large-v3">Large V3 (best)</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Device</label>
                <select
                  value={formData.default_device || 'auto'}
                  onChange={(e) => handleChange('default_device', e.target.value)}
                  className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                >
                  <option value="auto">Auto</option>
                  <option value="cpu">CPU</option>
                  <option value="cuda">CUDA (GPU)</option>
                  <option value="mps">MPS (Apple Silicon)</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Compute Type</label>
                <select
                  value={formData.default_compute_type || 'float16'}
                  onChange={(e) => handleChange('default_compute_type', e.target.value)}
                  className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                >
                  <option value="float32">Float32 (most accurate)</option>
                  <option value="float16">Float16 (balanced)</option>
                  <option value="int8">Int8 (fastest)</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Default Language</label>
                <Input
                  value={formData.default_language || ''}
                  onChange={(e) => handleChange('default_language', e.target.value)}
                  placeholder="e.g., en, es, fr (empty for auto-detect)"
                />
              </div>
            </div>
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="prefer_gpu"
                checked={formData.prefer_gpu || false}
                onChange={(e) => handleChange('prefer_gpu', e.target.checked)}
                className="rounded border-input"
              />
              <label htmlFor="prefer_gpu" className="text-sm">
                Prefer GPU when available
              </label>
            </div>
          </CardContent>
        </Card>

        {/* Translation Settings */}
        <Card>
          <CardHeader>
            <CardTitle>Translation</CardTitle>
            <CardDescription>Configure translation model defaults</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1">Translation Backend</label>
              <Input
                value={formData.default_translation_backend || ''}
                onChange={(e) =>
                  handleChange('default_translation_backend', e.target.value || null)
                }
                placeholder="e.g., facebook/nllb-200-distilled-600M"
              />
            </div>
          </CardContent>
        </Card>

        {/* Worker Settings */}
        <Card>
          <CardHeader>
            <CardTitle>Worker & Retention</CardTitle>
            <CardDescription>Configure job processing and data retention</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="block text-sm font-medium mb-1">Worker Count</label>
                <Input
                  type="number"
                  min="1"
                  max="10"
                  value={formData.worker_count || 2}
                  onChange={(e) => handleChange('worker_count', parseInt(e.target.value))}
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Log Retention (days)</label>
                <Input
                  type="number"
                  min="1"
                  value={formData.log_retention_days || 30}
                  onChange={(e) => handleChange('log_retention_days', parseInt(e.target.value))}
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Job Retention (days)</label>
                <Input
                  type="number"
                  min="1"
                  value={formData.job_retention_days || 90}
                  onChange={(e) => handleChange('job_retention_days', parseInt(e.target.value))}
                />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Save Button */}
        <div className="flex items-center gap-4">
          <Button type="submit" disabled={updateMutation.isPending}>
            {updateMutation.isPending ? (
              <>
                <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                Saving...
              </>
            ) : (
              <>
                <Save className="h-4 w-4 mr-2" />
                Save Settings
              </>
            )}
          </Button>
          {saveSuccess && (
            <div className="flex items-center gap-2 text-success">
              <CheckCircle className="h-4 w-4" />
              <span>Settings saved successfully</span>
            </div>
          )}
          {updateMutation.isError && (
            <div className="flex items-center gap-2 text-destructive">
              <AlertCircle className="h-4 w-4" />
              <span>Failed to save settings</span>
            </div>
          )}
        </div>
      </form>
    </div>
  );
}
