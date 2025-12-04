# CI/CD Workflows Documentation

This document describes the GitHub Actions workflows for the homelab-subtitle-service project.

## Overview

The project uses a multi-stage CI/CD pipeline with three main workflows:

1. **PR Validation** (`pr-validation.yml`) - Validates pull requests to `main`
2. **Main Build** (`main-build.yml`) - Builds Docker images on merge to `main`
3. **Release** (`release.yml`) - Publishes to Docker Hub when a version tag is pushed

## Workflow Details

### 1. PR Validation Workflow

**Trigger:** Pull requests targeting the `main` branch

**Purpose:** Ensures code quality and security before merging to main

**Jobs:**
- ✅ **Code Quality**: Ruff linting and formatting checks
- ✅ **Unit Tests**: pytest with coverage reporting (60% threshold)
- ✅ **Security Scanning**: Bandit (Python), Safety (dependencies)
- ✅ **Static Analysis**: Semgrep with auto-config rules
- ✅ **Docker Validation**: Dockerfile linting (Hadolint) and build test
- ✅ **Vulnerability Scanning**: Trivy for container and filesystem

**Behavior:**
- Blocks merge if critical issues found
- Generates PR summary with all validation results
- Uploads artifacts for detailed review

**Example PR Workflow:**
```bash
# Create feature branch
git checkout -b feature/new-subtitle-format

# Make changes and commit
git add .
git commit -m "Add support for WebVTT format"
git push origin feature/new-subtitle-format

# Create PR on GitHub
# → Triggers pr-validation.yml automatically
# → Review results in PR checks
# → Merge when all checks pass
```

### 2. Main Build Workflow

**Trigger:** Push to `main` branch (typically after PR merge)

**Purpose:** Builds production-ready Docker images and runs final validation

**Jobs:**
- ✅ **Re-validate**: Quick validation (tests, linting, security)
- ✅ **Build Image**: Creates Docker image with metadata
- ✅ **Security Scan**: Trivy scan on built image
- ✅ **Artifact Upload**: Stores image artifact (7 day retention)
- 🔧 **Self-Hosted Deploy** (optional, commented out by default)

**Image Naming Convention:**
```
homelab-subtitle-service:main-<short-sha>-<timestamp>
Example: homelab-subtitle-service:main-a1b2c3d-20251203-143022
```

**Behavior:**
- Runs comprehensive validation
- Builds Docker image but doesn't push to registry
- Stores image as artifact for testing/deployment
- Optional: Deploy to self-hosted runner (requires configuration)

**Example Main Build:**
```bash
# After PR is merged to main
# → Triggers main-build.yml automatically
# → Builds Docker image
# → Artifact available for download (7 days)
# → Ready for testing or deployment
```

### 3. Release Workflow

**Trigger:** Push of version tags (e.g., `v1.0.0`, `v2.1.3`)

**Purpose:** Publishes stable releases to Docker Hub

**Jobs:**
- ✅ **Validate Tag**: Extracts version, detects pre-releases
- ✅ **Full Test Suite**: Runs all tests and security scans
- ✅ **Multi-Platform Build**: Builds for linux/amd64 and linux/arm64
- ✅ **Push to Docker Hub**: Publishes to Docker Hub registry
- ✅ **Security Scan**: Trivy scan on published image
- ✅ **GitHub Release**: Creates GitHub release with changelog

**Tagging Strategy:**
```bash
# Stable release (tags: v1.0.0, 1.0, 1, latest)
git tag -a v1.0.0 -m "Release v1.0.0: Initial stable release"
git push origin v1.0.0

# Pre-release (tags: v1.1.0-beta.1, NO latest tag)
git tag -a v1.1.0-beta.1 -m "Beta release for testing"
git push origin v1.1.0-beta.1

# Patch release (tags: v1.0.1, 1.0, 1, latest)
git tag -a v1.0.1 -m "Release v1.0.1: Bug fixes"
git push origin v1.0.1
```

**Docker Hub Tags:**
- **Stable releases** (e.g., `v1.0.0`):
  - `1.0.0` (full version)
  - `1.0` (major.minor)
  - `1` (major)
  - `latest` (latest stable)

- **Pre-releases** (e.g., `v1.1.0-beta.1`):
  - `1.1.0-beta.1` (full version)
  - NO `latest` tag

**Behavior:**
- Runs full validation suite
- Builds for multiple platforms (amd64, arm64)
- Pushes to Docker Hub with semantic versioning
- Creates GitHub release with auto-generated notes
- Includes pull instructions in release

## Required GitHub Secrets

To use these workflows, configure the following secrets in your GitHub repository:

### Docker Hub Secrets

1. Go to: `Settings` → `Secrets and variables` → `Actions`
2. Add these secrets:

| Secret Name | Description | How to Get |
|------------|-------------|------------|
| `DOCKER_HUB_USERNAME` | Your Docker Hub username | Your Docker Hub account name |
| `DOCKER_HUB_TOKEN` | Docker Hub access token | [Create at Docker Hub](https://hub.docker.com/settings/security) |

**Creating Docker Hub Token:**
```bash
1. Log in to Docker Hub
2. Go to Account Settings → Security
3. Click "New Access Token"
4. Name: "GitHub Actions CI/CD"
5. Permissions: "Read, Write, Delete"
6. Copy the token (shown only once!)
7. Add to GitHub Secrets
```

## Self-Hosted Runner Setup (Optional)

To enable automatic deployment to your homelab:

### 1. Set Up Self-Hosted Runner

```bash
# On your homelab server
cd /opt
sudo mkdir actions-runner && cd actions-runner

# Download runner (check for latest version)
curl -o actions-runner-linux-x64-2.311.0.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz

# Extract
tar xzf ./actions-runner-linux-x64-2.311.0.tar.gz

# Configure (follow prompts)
./config.sh --url https://github.com/Jurel89/homelab-subtitle-service --token <YOUR_TOKEN>

# Install as service
sudo ./svc.sh install
sudo ./svc.sh start
```

### 2. Enable Deployment in main-build.yml

Uncomment the `deploy-selfhosted` job in `.github/workflows/main-build.yml`:

```yaml
deploy-selfhosted:
  name: Deploy to Self-Hosted Runner
  runs-on: self-hosted
  needs: build-image
  if: github.event_name == 'push' && github.ref == 'refs/heads/main'
  steps:
    # ... deployment steps ...
```

### 3. Configure Volumes

Update the deployment command to match your paths:

```yaml
docker run -d \
  --name homelab-subtitle-service \
  --restart unless-stopped \
  -v /path/to/your/videos:/videos \
  -v /path/to/your/subtitles:/output \
  ${{ needs.build-image.outputs.image-tag }}
```

## Usage Examples

### Standard Release Process

```bash
# 1. Develop on feature branch
git checkout -b feature/batch-processing
# ... make changes ...
git commit -am "Add batch processing support"
git push origin feature/batch-processing

# 2. Create PR on GitHub
# → pr-validation.yml runs automatically
# → Review checks and merge

# 3. After merge to main
# → main-build.yml runs automatically
# → Docker image built and available as artifact

# 4. Test the main branch build
# → Download artifact from GitHub Actions
# → Test in your environment

# 5. When ready to release
git checkout main
git pull origin main
git tag -a v1.1.0 -m "Release v1.1.0: Batch processing support"
git push origin v1.1.0

# → release.yml runs automatically
# → Image pushed to Docker Hub
# → GitHub release created
# → Ready for public use!
```

### Hotfix Release

```bash
# 1. Create hotfix branch from main
git checkout main
git checkout -b hotfix/critical-bug
# ... fix bug ...
git commit -am "Fix critical transcription bug"
git push origin hotfix/critical-bug

# 2. Create PR and merge quickly
# → pr-validation.yml runs
# → Merge to main

# 3. Create patch release immediately
git checkout main
git pull origin main
git tag -a v1.0.1 -m "Release v1.0.1: Critical bug fix"
git push origin v1.0.1

# → release.yml publishes to Docker Hub
```

### Beta/Pre-release

```bash
# 1. Develop new feature
git checkout -b feature/experimental-model
# ... implement ...
git commit -am "Add experimental Whisper v4 model"

# 2. Merge to main (after PR validation)

# 3. Create beta tag
git tag -a v1.2.0-beta.1 -m "Beta: Experimental model support"
git push origin v1.2.0-beta.1

# → Publishes to Docker Hub with version tag
# → NO 'latest' tag (safe for beta testing)
# → Users can test: docker pull user/image:1.2.0-beta.1
```

## Monitoring and Troubleshooting

### View Workflow Status

1. Go to `Actions` tab on GitHub
2. Select workflow from left sidebar
3. Click on specific run to see details
4. Download artifacts for logs/reports

### Common Issues

#### Issue: PR validation fails on Bandit

**Solution:** Check Bandit report artifact, add `# nosec` comments with justification if false positive

#### Issue: Docker build fails on Hadolint

**Solution:** Review Hadolint SARIF report, fix Dockerfile issues (version pinning, best practices)

#### Issue: Trivy finds HIGH/CRITICAL vulnerabilities

**Solution:**
1. Check if vulnerabilities affect your use case
2. Update base image or dependencies
3. If unavoidable, document in security notes

#### Issue: Coverage drops below threshold

**Solution:** Add more unit tests or adjust threshold in `pr-validation.yml`

#### Issue: Self-hosted deployment fails

**Solution:**
1. Check runner is online: `sudo ./svc.sh status`
2. Verify runner has Docker access: `sudo usermod -aG docker $(whoami)`
3. Check runner logs: `journalctl -u actions.runner.*`

### Workflow Logs

All workflows generate detailed summaries available in:
- GitHub Actions UI (Summary tab)
- Downloadable artifacts (reports, coverage, etc.)
- GitHub Releases (for release workflow)

## Maintenance

### Update Workflows

When updating workflows, consider:
1. Test changes in a feature branch first
2. Review security implications
3. Update this documentation
4. Test with a pre-release tag before stable release

### Security Best Practices

1. **Never commit secrets** to repository
2. **Rotate Docker Hub tokens** periodically
3. **Review Dependabot alerts** regularly
4. **Keep runners updated** (self-hosted)
5. **Monitor workflow permissions** (least privilege)

## Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Hub Documentation](https://docs.docker.com/docker-hub/)
- [Semantic Versioning](https://semver.org/)
- [Self-Hosted Runners Guide](https://docs.github.com/en/actions/hosting-your-own-runners)

## Support

For issues or questions about CI/CD:
1. Check workflow logs in GitHub Actions
2. Review this documentation
3. Open an issue on GitHub
4. Check [GitHub Discussions](https://github.com/Jurel89/homelab-subtitle-service/discussions)
