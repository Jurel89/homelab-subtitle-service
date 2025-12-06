import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Folder, File, ChevronRight, ChevronUp } from 'lucide-react';
import { jobsApi, filesApi, settingsApi } from '@/lib/api';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/Card';
import type { JobType, JobCreateRequest, FileItem, GlobalSettings } from '@/types';

const STORAGE_KEY = 'subsvc_last_job_options';

interface JobOptions {
  type: JobType;
  model_size: string;
  compute_type: string;
  source_language: string;
  target_language: string;
  prefer_gpu: boolean;
}

const defaultOptions: JobOptions = {
  type: 'transcribe',
  model_size: 'base',
  compute_type: 'float16',
  source_language: '',
  target_language: '',
  prefer_gpu: true,
};

export function NewJobPage() {
  const navigate = useNavigate();
  const [options, setOptions] = useState<JobOptions>(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    return saved ? JSON.parse(saved) : defaultOptions;
  });
  const [inputPath, setInputPath] = useState('');
  const [referencePath, setReferencePath] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // File browser state
  const [showFileBrowser, setShowFileBrowser] = useState(false);
  const [browserTarget, setBrowserTarget] = useState<'input' | 'reference'>('input');
  const [currentPath, setCurrentPath] = useState<string | null>(null);
  const [files, setFiles] = useState<FileItem[]>([]);
  const [parentPath, setParentPath] = useState<string | null>(null);
  const [isLoadingFiles, setIsLoadingFiles] = useState(false);
  
  // Settings (for future use with defaults)
  const [, setSettings] = useState<GlobalSettings | null>(null);

  // Load settings on mount
  useEffect(() => {
    settingsApi.get().then(setSettings).catch(console.error);
  }, []);

  // Save options to localStorage
  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(options));
  }, [options]);

  // Load files when browser is opened or path changes
  useEffect(() => {
    if (!showFileBrowser) return;
    
    const loadFiles = async () => {
      setIsLoadingFiles(true);
      try {
        const res = await filesApi.browse(currentPath || undefined);
        setFiles(res.items);
        setParentPath(res.parent_path);
      } catch (err) {
        console.error('Failed to load files:', err);
        setFiles([]);
      } finally {
        setIsLoadingFiles(false);
      }
    };
    
    loadFiles();
  }, [showFileBrowser, currentPath]);

  const handleFileSelect = (item: FileItem) => {
    if (item.is_directory) {
      setCurrentPath(item.path);
    } else {
      if (browserTarget === 'input') {
        setInputPath(item.path);
      } else {
        setReferencePath(item.path);
      }
      setShowFileBrowser(false);
    }
  };

  const openFileBrowser = (target: 'input' | 'reference') => {
    setBrowserTarget(target);
    setCurrentPath(null);
    setShowFileBrowser(true);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!inputPath) {
      setError('Please select an input file');
      return;
    }

    setIsSubmitting(true);

    try {
      const request: JobCreateRequest = {
        type: options.type,
        input_path: inputPath,
        model_size: options.model_size,
        compute_type: options.compute_type,
        source_language: options.source_language || undefined,
        target_language: options.target_language || undefined,
        reference_path: referencePath || undefined,
        options: {
          prefer_gpu: options.prefer_gpu,
        },
      };

      const job = await jobsApi.create(request);
      navigate(`/jobs/${job.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create job');
    } finally {
      setIsSubmitting(false);
    }
  };

  const needsReference = options.type === 'sync' || options.type === 'compare';
  const needsTargetLanguage = options.type === 'translate';

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Create New Job</h1>
        <p className="text-muted-foreground">Configure and submit a new subtitle processing job</p>
      </div>

      {error && (
        <div className="rounded-md bg-destructive/10 p-4 text-destructive">
          {error}
        </div>
      )}

      <form onSubmit={handleSubmit}>
        <div className="grid gap-6 lg:grid-cols-2">
          {/* Job Type */}
          <Card>
            <CardHeader>
              <CardTitle>Job Type</CardTitle>
              <CardDescription>Select the type of processing to perform</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-2">
                {(['transcribe', 'translate', 'sync', 'compare'] as JobType[]).map((type) => (
                  <button
                    key={type}
                    type="button"
                    onClick={() => setOptions({ ...options, type })}
                    className={`rounded-md border p-3 text-left transition-colors ${
                      options.type === type
                        ? 'border-primary bg-primary/10'
                        : 'border-input hover:bg-accent'
                    }`}
                  >
                    <div className="font-medium capitalize">{type}</div>
                    <div className="text-xs text-muted-foreground">
                      {type === 'transcribe' && 'Generate subtitles from audio'}
                      {type === 'translate' && 'Translate existing subtitles'}
                      {type === 'sync' && 'Sync subtitles to audio'}
                      {type === 'compare' && 'Compare subtitle accuracy'}
                    </div>
                  </button>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Model Settings */}
          <Card>
            <CardHeader>
              <CardTitle>Model Settings</CardTitle>
              <CardDescription>Configure the AI model parameters</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="mb-2 block text-sm font-medium">Model Size</label>
                <select
                  className="w-full rounded-md border bg-background px-3 py-2"
                  value={options.model_size}
                  onChange={(e) => setOptions({ ...options, model_size: e.target.value })}
                >
                  <option value="tiny">Tiny (fastest)</option>
                  <option value="base">Base</option>
                  <option value="small">Small</option>
                  <option value="medium">Medium</option>
                  <option value="large">Large (most accurate)</option>
                </select>
              </div>
              <div>
                <label className="mb-2 block text-sm font-medium">Compute Type</label>
                <select
                  className="w-full rounded-md border bg-background px-3 py-2"
                  value={options.compute_type}
                  onChange={(e) => setOptions({ ...options, compute_type: e.target.value })}
                >
                  <option value="float16">Float16 (GPU)</option>
                  <option value="float32">Float32</option>
                  <option value="int8">Int8 (CPU optimized)</option>
                </select>
              </div>
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="prefer_gpu"
                  checked={options.prefer_gpu}
                  onChange={(e) => setOptions({ ...options, prefer_gpu: e.target.checked })}
                  className="h-4 w-4 rounded border-input"
                />
                <label htmlFor="prefer_gpu" className="text-sm">
                  Prefer GPU when available
                </label>
              </div>
            </CardContent>
          </Card>

          {/* Language Settings */}
          <Card>
            <CardHeader>
              <CardTitle>Language</CardTitle>
              <CardDescription>Source and target language settings</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="mb-2 block text-sm font-medium">Source Language</label>
                <Input
                  placeholder="Auto-detect (leave empty)"
                  value={options.source_language}
                  onChange={(e) => setOptions({ ...options, source_language: e.target.value })}
                />
                <p className="mt-1 text-xs text-muted-foreground">
                  e.g., en, es, fr, de, ja, zh
                </p>
              </div>
              {needsTargetLanguage && (
                <div>
                  <label className="mb-2 block text-sm font-medium">Target Language</label>
                  <Input
                    placeholder="Required for translation"
                    value={options.target_language}
                    onChange={(e) => setOptions({ ...options, target_language: e.target.value })}
                  />
                </div>
              )}
            </CardContent>
          </Card>

          {/* File Selection */}
          <Card>
            <CardHeader>
              <CardTitle>Files</CardTitle>
              <CardDescription>Select input and reference files</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="mb-2 block text-sm font-medium">Input File</label>
                <div className="flex space-x-2">
                  <Input
                    placeholder="Select a file..."
                    value={inputPath}
                    onChange={(e) => setInputPath(e.target.value)}
                    className="flex-1"
                  />
                  <Button type="button" variant="outline" onClick={() => openFileBrowser('input')}>
                    <Folder className="h-4 w-4" />
                  </Button>
                </div>
              </div>
              {needsReference && (
                <div>
                  <label className="mb-2 block text-sm font-medium">Reference File</label>
                  <div className="flex space-x-2">
                    <Input
                      placeholder="Select a reference file..."
                      value={referencePath}
                      onChange={(e) => setReferencePath(e.target.value)}
                      className="flex-1"
                    />
                    <Button type="button" variant="outline" onClick={() => openFileBrowser('reference')}>
                      <Folder className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Submit */}
        <div className="mt-6 flex justify-end space-x-2">
          <Button type="button" variant="outline" onClick={() => navigate('/')}>
            Cancel
          </Button>
          <Button type="submit" isLoading={isSubmitting}>
            Create Job
          </Button>
        </div>
      </form>

      {/* File Browser Modal */}
      {showFileBrowser && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <Card className="m-4 max-h-[80vh] w-full max-w-2xl overflow-hidden">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Select File</CardTitle>
                <Button variant="ghost" size="sm" onClick={() => setShowFileBrowser(false)}>
                  ×
                </Button>
              </div>
              <CardDescription className="truncate">
                {currentPath || 'Media Folders'}
              </CardDescription>
            </CardHeader>
            <CardContent className="max-h-96 overflow-auto">
              {parentPath && (
                <button
                  className="mb-2 flex w-full items-center space-x-2 rounded-md p-2 text-left hover:bg-accent"
                  onClick={() => setCurrentPath(parentPath === '/' ? null : parentPath)}
                >
                  <ChevronUp className="h-4 w-4" />
                  <span>..</span>
                </button>
              )}
              {isLoadingFiles ? (
                <div className="flex justify-center py-8">
                  <div className="h-6 w-6 animate-spin rounded-full border-2 border-primary border-t-transparent" />
                </div>
              ) : files.length === 0 ? (
                <div className="py-8 text-center text-muted-foreground">
                  No files found
                </div>
              ) : (
                <div className="space-y-1">
                  {files.map((item) => (
                    <button
                      key={item.path}
                      className="flex w-full items-center space-x-2 rounded-md p-2 text-left hover:bg-accent"
                      onClick={() => handleFileSelect(item)}
                    >
                      {item.is_directory ? (
                        <Folder className="h-4 w-4 text-primary" />
                      ) : (
                        <File className="h-4 w-4" />
                      )}
                      <span className="flex-1 truncate">{item.name}</span>
                      {item.is_directory && <ChevronRight className="h-4 w-4" />}
                    </button>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
