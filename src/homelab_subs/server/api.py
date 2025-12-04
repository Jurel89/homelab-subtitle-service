"""
FastAPI REST API for the Homelab Subtitle Service.

This module provides the HTTP API for managing subtitle generation jobs,
including job creation, status queries, cancellation, and retry operations.
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, ConfigDict

from homelab_subs.server.settings import get_settings, Settings
from homelab_subs.server.models import JobStatus, JobType
from homelab_subs.server.job_service import ServerJobService

logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Schemas for Request/Response
# =============================================================================


class JobCreateRequest(BaseModel):
    """Request schema for creating a new job."""

    type: JobType = Field(
        default=JobType.TRANSCRIBE, description="Type of job to create"
    )
    input_path: str = Field(
        ..., description="Path to input file (video, audio, or SRT)"
    )
    output_path: Optional[str] = Field(
        default=None,
        description="Path for output file (auto-generated if not specified)",
    )
    reference_path: Optional[str] = Field(
        default=None, description="Path to reference file (for sync or compare jobs)"
    )
    source_language: Optional[str] = Field(
        default=None, description="Source language code (e.g., 'en', 'es')"
    )
    target_language: Optional[str] = Field(
        default=None, description="Target language code for translation"
    )
    model_size: str = Field(
        default="base",
        description="Whisper model size: tiny, base, small, medium, large",
    )
    compute_type: str = Field(
        default="float16", description="Compute precision: float16, float32, int8"
    )
    priority: str = Field(
        default="default", description="Job priority: high, default, low"
    )
    options: Optional[dict] = Field(
        default=None, description="Additional job-specific options"
    )

    model_config = ConfigDict(use_enum_values=True)


class JobResponse(BaseModel):
    """Response schema for job details."""

    id: UUID
    type: str
    status: str
    stage: str
    progress: int
    input_path: str
    output_path: Optional[str]
    reference_path: Optional[str]
    source_language: Optional[str]
    target_language: Optional[str]
    model_size: Optional[str]
    compute_type: Optional[str]
    error_message: Optional[str]
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]

    model_config = ConfigDict(from_attributes=True)


class JobListResponse(BaseModel):
    """Response schema for job listing."""

    jobs: list[JobResponse]
    total: int
    page: int
    page_size: int
    has_more: bool


class JobLogsResponse(BaseModel):
    """Response schema for job logs."""

    job_id: UUID
    logs: Optional[str]
    status: str
    stage: str


class JobStatisticsResponse(BaseModel):
    """Response schema for job statistics."""

    total_jobs: int
    pending: int
    running: int
    completed: int
    failed: int
    cancelled: int


class QueueStatusResponse(BaseModel):
    """Response schema for queue status."""

    queues: dict
    workers: list
    total_jobs: int
    failed_jobs: int


class HealthResponse(BaseModel):
    """Response schema for health check."""

    status: str
    database: str
    redis: str
    version: str


class ErrorResponse(BaseModel):
    """Response schema for errors."""

    detail: str
    error_code: Optional[str] = None


# =============================================================================
# Dependencies
# =============================================================================


def get_settings_dep() -> Settings:
    """Dependency for settings."""
    return get_settings()


async def get_job_service(
    settings: Settings = Depends(get_settings_dep),
) -> ServerJobService:
    """Dependency for job service."""
    service = ServerJobService(settings)
    return service


# =============================================================================
# Lifespan and App Setup
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting Homelab Subtitle Service API")
    yield
    logger.info("Shutting down Homelab Subtitle Service API")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Homelab Subtitle Service",
        description="API for managing subtitle generation jobs with Whisper transcription, translation, and synchronization",
        version="0.3.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


app = create_app()


# =============================================================================
# Health & Status Endpoints
# =============================================================================


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(service: ServerJobService = Depends(get_job_service)):
    """
    Health check endpoint.

    Returns the health status of the API, database, and Redis connections.
    """
    db_status = "unknown"
    redis_status = "unknown"

    try:
        # Check database
        await service.repository.get_statistics()
        db_status = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = "unhealthy"

    try:
        # Check Redis
        service.queue_client.get_queue_status()
        redis_status = "healthy"
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        redis_status = "unhealthy"

    overall_status = (
        "healthy"
        if db_status == "healthy" and redis_status == "healthy"
        else "degraded"
    )

    return HealthResponse(
        status=overall_status,
        database=db_status,
        redis=redis_status,
        version="0.3.0",
    )


@app.get("/stats", response_model=JobStatisticsResponse, tags=["System"])
async def get_statistics(service: ServerJobService = Depends(get_job_service)):
    """
    Get job statistics.

    Returns counts of jobs by status.
    """
    stats = await service.get_statistics()
    return JobStatisticsResponse(**stats)


@app.get("/queue/status", response_model=QueueStatusResponse, tags=["System"])
async def get_queue_status(service: ServerJobService = Depends(get_job_service)):
    """
    Get queue status.

    Returns information about job queues and workers.
    """
    status = service.get_queue_status()
    return QueueStatusResponse(**status)


# =============================================================================
# Job CRUD Endpoints
# =============================================================================


@app.post(
    "/jobs",
    response_model=JobResponse,
    status_code=201,
    tags=["Jobs"],
    responses={400: {"model": ErrorResponse}},
)
async def create_job(
    request: JobCreateRequest,
    service: ServerJobService = Depends(get_job_service),
):
    """
    Create a new subtitle processing job.

    The job will be queued for processing by a worker. Use the returned
    job ID to track progress or cancel the job.

    Job types:
    - **transcribe**: Generate subtitles from audio/video
    - **translate**: Translate existing subtitles to another language
    - **sync**: Synchronize subtitles with audio
    - **compare**: Compare two subtitle files for accuracy metrics
    - **full_pipeline**: Complete workflow with optional translation
    """
    # Validate input path exists
    input_path = Path(request.input_path)
    if not input_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Input file not found: {request.input_path}",
        )

    # Validate reference path for jobs that need it
    if request.type in [JobType.SYNC, JobType.COMPARE]:
        if not request.reference_path:
            raise HTTPException(
                status_code=400,
                detail=f"Reference path is required for {request.type} jobs",
            )
        if not Path(request.reference_path).exists():
            raise HTTPException(
                status_code=400,
                detail=f"Reference file not found: {request.reference_path}",
            )

    # Validate translation has target language
    if request.type == JobType.TRANSLATE and not request.target_language:
        raise HTTPException(
            status_code=400,
            detail="Target language is required for translation jobs",
        )

    try:
        job = await service.create_job(
            job_type=request.type,
            input_path=request.input_path,
            output_path=request.output_path,
            reference_path=request.reference_path,
            source_language=request.source_language,
            target_language=request.target_language,
            model_size=request.model_size,
            compute_type=request.compute_type,
            priority=request.priority,
            options=request.options,
        )
        return _job_to_response(job)
    except Exception as e:
        logger.exception("Failed to create job")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/jobs", response_model=JobListResponse, tags=["Jobs"])
async def list_jobs(
    status: Optional[JobStatus] = Query(default=None, description="Filter by status"),
    job_type: Optional[JobType] = Query(
        default=None, alias="type", description="Filter by type"
    ),
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
    service: ServerJobService = Depends(get_job_service),
):
    """
    List jobs with optional filtering and pagination.

    Returns a paginated list of jobs ordered by creation date (newest first).
    """
    offset = (page - 1) * page_size

    jobs, total = await service.list_jobs(
        status=status,
        job_type=job_type,
        limit=page_size + 1,  # Fetch one extra to check for more
        offset=offset,
    )

    has_more = len(jobs) > page_size
    if has_more:
        jobs = jobs[:page_size]

    return JobListResponse(
        jobs=[_job_to_response(job) for job in jobs],
        total=total,
        page=page,
        page_size=page_size,
        has_more=has_more,
    )


@app.get(
    "/jobs/{job_id}",
    response_model=JobResponse,
    tags=["Jobs"],
    responses={404: {"model": ErrorResponse}},
)
async def get_job(
    job_id: UUID,
    service: ServerJobService = Depends(get_job_service),
):
    """
    Get details of a specific job.

    Returns full job information including progress, stage, and any errors.
    """
    job = await service.get_job(str(job_id))
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    return _job_to_response(job)


@app.get(
    "/jobs/{job_id}/logs",
    response_model=JobLogsResponse,
    tags=["Jobs"],
    responses={404: {"model": ErrorResponse}},
)
async def get_job_logs(
    job_id: UUID,
    service: ServerJobService = Depends(get_job_service),
):
    """
    Get logs for a specific job.

    Returns the accumulated logs from job processing.
    """
    job = await service.get_job(str(job_id))
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    return JobLogsResponse(
        job_id=job_id,
        logs=job.logs,
        status=job.status.value if hasattr(job.status, "value") else str(job.status),
        stage=job.stage.value if hasattr(job.stage, "value") else str(job.stage),
    )


@app.get(
    "/jobs/{job_id}/output",
    tags=["Jobs"],
    responses={
        404: {"model": ErrorResponse},
        400: {"model": ErrorResponse},
    },
)
async def download_output(
    job_id: UUID,
    service: ServerJobService = Depends(get_job_service),
):
    """
    Download the output file of a completed job.

    Returns the generated SRT file or comparison report.
    """
    job = await service.get_job(str(job_id))
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed (status: {job.status})",
        )

    if not job.output_path:
        raise HTTPException(
            status_code=400,
            detail="Job has no output file",
        )

    output_path = Path(job.output_path)
    if not output_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Output file not found: {job.output_path}",
        )

    return FileResponse(
        path=output_path,
        filename=output_path.name,
        media_type="application/octet-stream",
    )


# =============================================================================
# Job Actions
# =============================================================================


@app.post(
    "/jobs/{job_id}/cancel",
    response_model=JobResponse,
    tags=["Jobs"],
    responses={
        404: {"model": ErrorResponse},
        400: {"model": ErrorResponse},
    },
)
async def cancel_job(
    job_id: UUID,
    service: ServerJobService = Depends(get_job_service),
):
    """
    Cancel a pending or running job.

    Cancellation is best-effort. Running jobs will complete their current
    stage before stopping.
    """
    try:
        job = await service.cancel_job(str(job_id))
        return _job_to_response(job)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post(
    "/jobs/{job_id}/retry",
    response_model=JobResponse,
    tags=["Jobs"],
    responses={
        404: {"model": ErrorResponse},
        400: {"model": ErrorResponse},
    },
)
async def retry_job(
    job_id: UUID,
    service: ServerJobService = Depends(get_job_service),
):
    """
    Retry a failed or cancelled job.

    Creates a new job with the same parameters and queues it for processing.
    """
    try:
        job = await service.retry_job(str(job_id))
        return _job_to_response(job)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete(
    "/jobs/{job_id}",
    status_code=204,
    tags=["Jobs"],
    responses={
        404: {"model": ErrorResponse},
        400: {"model": ErrorResponse},
    },
)
async def delete_job(
    job_id: UUID,
    service: ServerJobService = Depends(get_job_service),
):
    """
    Delete a job.

    Only completed, failed, or cancelled jobs can be deleted.
    """
    job = await service.get_job(str(job_id))
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if job.status in [JobStatus.PENDING, JobStatus.RUNNING]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot delete job with status: {job.status}. Cancel it first.",
        )

    await service.repository.delete(str(job_id))
    return None


# =============================================================================
# Batch Operations
# =============================================================================


@app.post(
    "/jobs/batch",
    response_model=list[JobResponse],
    status_code=201,
    tags=["Jobs"],
    responses={400: {"model": ErrorResponse}},
)
async def create_batch_jobs(
    requests: list[JobCreateRequest],
    service: ServerJobService = Depends(get_job_service),
):
    """
    Create multiple jobs in a batch.

    All jobs are validated and created atomically. If any job fails
    validation, none are created.
    """
    if len(requests) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 jobs per batch",
        )

    # Validate all jobs first
    for i, req in enumerate(requests):
        if not Path(req.input_path).exists():
            raise HTTPException(
                status_code=400,
                detail=f"Job {i}: Input file not found: {req.input_path}",
            )

    # Create all jobs
    jobs = []
    for req in requests:
        job = await service.create_job(
            job_type=req.type,
            input_path=req.input_path,
            output_path=req.output_path,
            reference_path=req.reference_path,
            source_language=req.source_language,
            target_language=req.target_language,
            model_size=req.model_size,
            compute_type=req.compute_type,
            priority=req.priority,
            options=req.options,
        )
        jobs.append(job)

    return [_job_to_response(job) for job in jobs]


# =============================================================================
# Helper Functions
# =============================================================================


def _job_to_response(job) -> JobResponse:
    """Convert a Job model to a JobResponse."""
    return JobResponse(
        id=job.id,
        type=job.type.value if hasattr(job.type, "value") else str(job.type),
        status=job.status.value if hasattr(job.status, "value") else str(job.status),
        stage=job.stage.value if hasattr(job.stage, "value") else str(job.stage),
        progress=job.progress,
        input_path=job.input_path,
        output_path=job.output_path,
        reference_path=job.reference_path,
        source_language=job.source_language,
        target_language=job.target_language,
        model_size=job.model_size,
        compute_type=job.compute_type,
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


# =============================================================================
# Error Handlers
# =============================================================================


@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle ValueError exceptions."""
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions."""
    logger.exception("Unexpected error in API")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
