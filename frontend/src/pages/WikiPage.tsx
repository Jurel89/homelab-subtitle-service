import { useState } from 'react';
import { 
  Book, 
  ChevronRight, 
  Code, 
  FileVideo, 
  Languages, 
  Settings, 
  Zap,
  Terminal,
  Server,
  Database,
  Workflow
} from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';

interface DocSection {
  id: string;
  title: string;
  icon: React.ReactNode;
  content: React.ReactNode;
}

const sections: DocSection[] = [
  {
    id: 'getting-started',
    title: 'Getting Started',
    icon: <Zap className="h-5 w-5" />,
    content: (
      <div className="prose prose-invert max-w-none">
        <h3>Quick Start Guide</h3>
        <p>HomeLab Subtitle Service is a self-hosted solution for automated subtitle generation, translation, and synchronization.</p>
        
        <h4>Installation</h4>
        <pre className="bg-muted p-4 rounded-lg overflow-x-auto">
{`# Clone the repository
git clone https://github.com/yourusername/homelab-subtitle-service
cd homelab-subtitle-service

# Install dependencies
pip install -e ".[dev]"

# Start the server
homelab-subs server start`}
        </pre>

        <h4>First Steps</h4>
        <ol>
          <li>Start the server and worker processes</li>
          <li>Access the web UI at <code>http://localhost:8000</code></li>
          <li>Create your first job using the "New Job" button</li>
          <li>Monitor progress in real-time from the dashboard</li>
        </ol>
      </div>
    ),
  },
  {
    id: 'job-types',
    title: 'Job Types',
    icon: <FileVideo className="h-5 w-5" />,
    content: (
      <div className="prose prose-invert max-w-none">
        <h3>Available Job Types</h3>
        
        <h4>Full Pipeline</h4>
        <p>Complete workflow: audio extraction → transcription → translation → synchronization. Best for video files without subtitles.</p>
        
        <h4>Transcription Only</h4>
        <p>Extract audio and generate subtitles using Whisper. No translation or sync.</p>
        
        <h4>Translation Only</h4>
        <p>Translate existing SRT files to another language using NLLB models.</p>
        
        <h4>Sync Only</h4>
        <p>Synchronize existing subtitles with audio/video using advanced alignment algorithms.</p>
        
        <h4>Compare Subtitles</h4>
        <p>Compare two SRT files and generate a detailed comparison report with accuracy metrics.</p>
      </div>
    ),
  },
  {
    id: 'models',
    title: 'AI Models',
    icon: <Languages className="h-5 w-5" />,
    content: (
      <div className="prose prose-invert max-w-none">
        <h3>Supported AI Models</h3>
        
        <h4>Whisper (Transcription)</h4>
        <table className="w-full">
          <thead>
            <tr>
              <th>Model</th>
              <th>Size</th>
              <th>Speed</th>
              <th>Quality</th>
            </tr>
          </thead>
          <tbody>
            <tr><td>tiny</td><td>39M</td><td>Fastest</td><td>Low</td></tr>
            <tr><td>base</td><td>74M</td><td>Fast</td><td>Good</td></tr>
            <tr><td>small</td><td>244M</td><td>Medium</td><td>Better</td></tr>
            <tr><td>medium</td><td>769M</td><td>Slow</td><td>High</td></tr>
            <tr><td>large-v3</td><td>1.5G</td><td>Slowest</td><td>Best</td></tr>
          </tbody>
        </table>

        <h4>NLLB (Translation)</h4>
        <p>Meta's No Language Left Behind model supports 200+ languages. Recommended model: <code>facebook/nllb-200-distilled-600M</code></p>
        
        <h4>Device Support</h4>
        <ul>
          <li><strong>CPU</strong>: Works everywhere, slower</li>
          <li><strong>CUDA</strong>: NVIDIA GPUs, fastest</li>
          <li><strong>MPS</strong>: Apple Silicon, good performance</li>
        </ul>
      </div>
    ),
  },
  {
    id: 'cli',
    title: 'CLI Reference',
    icon: <Terminal className="h-5 w-5" />,
    content: (
      <div className="prose prose-invert max-w-none">
        <h3>Command Line Interface</h3>
        
        <h4>Server Commands</h4>
        <pre className="bg-muted p-4 rounded-lg overflow-x-auto">
{`# Start the API server
homelab-subs server start --host 0.0.0.0 --port 8000

# Start a worker process
homelab-subs server worker --queues default high

# Check server status
homelab-subs server status`}
        </pre>

        <h4>Pipeline Commands</h4>
        <pre className="bg-muted p-4 rounded-lg overflow-x-auto">
{`# Run full pipeline
homelab-subs pipeline run video.mp4 --output subs.srt

# Transcribe only
homelab-subs transcribe video.mp4 --model large-v3

# Translate subtitles
homelab-subs translate input.srt --target spa_Latn

# Sync subtitles
homelab-subs sync video.mp4 subs.srt --output synced.srt

# Compare subtitles
homelab-subs compare reference.srt generated.srt`}
        </pre>
      </div>
    ),
  },
  {
    id: 'api',
    title: 'API Reference',
    icon: <Code className="h-5 w-5" />,
    content: (
      <div className="prose prose-invert max-w-none">
        <h3>REST API</h3>
        
        <h4>Authentication</h4>
        <pre className="bg-muted p-4 rounded-lg overflow-x-auto">
{`POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "admin",
  "password": "your-password"
}

Response:
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "token_type": "bearer"
}`}
        </pre>

        <h4>Jobs</h4>
        <pre className="bg-muted p-4 rounded-lg overflow-x-auto">
{`# Create a job
POST /api/v1/jobs
Authorization: Bearer <token>

{
  "job_type": "full_pipeline",
  "input_file": "/path/to/video.mp4",
  "options": {
    "whisper_model": "large-v3",
    "target_language": "spa_Latn"
  }
}

# List jobs
GET /api/v1/jobs?page=1&per_page=20

# Get job details
GET /api/v1/jobs/{job_id}

# Cancel job
POST /api/v1/jobs/{job_id}/cancel`}
        </pre>
      </div>
    ),
  },
  {
    id: 'architecture',
    title: 'Architecture',
    icon: <Workflow className="h-5 w-5" />,
    content: (
      <div className="prose prose-invert max-w-none">
        <h3>System Architecture</h3>
        
        <h4>Components</h4>
        <ul>
          <li><strong>API Server</strong>: FastAPI-based REST API</li>
          <li><strong>Job Queue</strong>: Redis + RQ for background processing</li>
          <li><strong>Workers</strong>: Process jobs from the queue</li>
          <li><strong>Database</strong>: PostgreSQL for job persistence</li>
          <li><strong>Frontend</strong>: React + TypeScript SPA</li>
        </ul>

        <h4>Data Flow</h4>
        <pre className="bg-muted p-4 rounded-lg overflow-x-auto">
{`Client → API Server → Job Queue → Worker → Pipeline
                ↓                        ↓
            Database ←────────────────────`}
        </pre>

        <h4>Pipeline Stages</h4>
        <ol>
          <li><strong>Audio Extraction</strong>: FFmpeg extracts audio from video</li>
          <li><strong>Transcription</strong>: Whisper generates subtitles</li>
          <li><strong>Translation</strong>: NLLB translates to target language</li>
          <li><strong>Synchronization</strong>: Align subtitles with audio</li>
        </ol>
      </div>
    ),
  },
  {
    id: 'configuration',
    title: 'Configuration',
    icon: <Settings className="h-5 w-5" />,
    content: (
      <div className="prose prose-invert max-w-none">
        <h3>Configuration Options</h3>
        
        <h4>Environment Variables</h4>
        <pre className="bg-muted p-4 rounded-lg overflow-x-auto">
{`# Server
HOMELAB_SUBS_HOST=0.0.0.0
HOMELAB_SUBS_PORT=8000

# Database
DATABASE_URL=postgresql://user:pass@localhost/homelab_subs

# Redis
REDIS_URL=redis://localhost:6379

# Security
JWT_SECRET_KEY=your-secret-key
JWT_ALGORITHM=HS256

# Models
WHISPER_MODEL=large-v3
WHISPER_DEVICE=auto
TRANSLATION_MODEL=facebook/nllb-200-distilled-600M`}
        </pre>

        <h4>Settings Page</h4>
        <p>Use the Settings page in the web UI to configure default options for all job types without editing environment variables.</p>
      </div>
    ),
  },
  {
    id: 'deployment',
    title: 'Deployment',
    icon: <Server className="h-5 w-5" />,
    content: (
      <div className="prose prose-invert max-w-none">
        <h3>Deployment Options</h3>
        
        <h4>Docker</h4>
        <pre className="bg-muted p-4 rounded-lg overflow-x-auto">
{`# Build the image
docker build -t homelab-subs .

# Run with docker-compose
docker-compose up -d`}
        </pre>

        <h4>Docker Compose</h4>
        <pre className="bg-muted p-4 rounded-lg overflow-x-auto">
{`version: '3.8'
services:
  api:
    image: homelab-subs
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db/homelab_subs
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis

  worker:
    image: homelab-subs
    command: homelab-subs server worker
    depends_on:
      - redis

  db:
    image: postgres:15
    environment:
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=homelab_subs

  redis:
    image: redis:7-alpine`}
        </pre>
      </div>
    ),
  },
  {
    id: 'database',
    title: 'Database',
    icon: <Database className="h-5 w-5" />,
    content: (
      <div className="prose prose-invert max-w-none">
        <h3>Database Schema</h3>
        
        <h4>Tables</h4>
        <ul>
          <li><strong>users</strong>: User accounts and authentication</li>
          <li><strong>jobs</strong>: Job records and status</li>
          <li><strong>global_settings</strong>: Application-wide settings</li>
        </ul>

        <h4>Migrations</h4>
        <pre className="bg-muted p-4 rounded-lg overflow-x-auto">
{`# Run migrations (if using Alembic)
alembic upgrade head

# Create a new migration
alembic revision --autogenerate -m "description"`}
        </pre>

        <h4>Backup</h4>
        <pre className="bg-muted p-4 rounded-lg overflow-x-auto">
{`# Backup database
pg_dump homelab_subs > backup.sql

# Restore database
psql homelab_subs < backup.sql`}
        </pre>
      </div>
    ),
  },
];

export default function WikiPage() {
  const [activeSection, setActiveSection] = useState(sections[0].id);

  const currentSection = sections.find((s) => s.id === activeSection) || sections[0];

  return (
    <div className="flex gap-6 h-[calc(100vh-8rem)]">
      {/* Sidebar Navigation */}
      <Card className="w-64 flex-shrink-0 h-full">
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-lg">
            <Book className="h-5 w-5" />
            Documentation
          </CardTitle>
        </CardHeader>
        <CardContent className="p-2">
          <nav className="space-y-1">
            {sections.map((section) => (
              <button
                key={section.id}
                onClick={() => setActiveSection(section.id)}
                className={`w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors text-left ${
                  activeSection === section.id
                    ? 'bg-accent text-accent-foreground'
                    : 'hover:bg-muted text-muted-foreground hover:text-foreground'
                }`}
              >
                {section.icon}
                <span className="flex-1">{section.title}</span>
                {activeSection === section.id && (
                  <ChevronRight className="h-4 w-4" />
                )}
              </button>
            ))}
          </nav>
        </CardContent>
      </Card>

      {/* Content Area */}
      <Card className="flex-1 h-full overflow-auto">
        <CardHeader className="border-b">
          <CardTitle className="flex items-center gap-3">
            {currentSection.icon}
            {currentSection.title}
          </CardTitle>
        </CardHeader>
        <CardContent className="p-6">
          {currentSection.content}
        </CardContent>
      </Card>
    </div>
  );
}
