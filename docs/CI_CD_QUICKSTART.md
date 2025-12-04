# Quick Setup Guide for CI/CD

Follow these steps to enable the complete CI/CD pipeline for homelab-subtitle-service.

## Prerequisites

- GitHub repository with admin access
- Docker Hub account (for publishing releases)

## Step 1: Configure Docker Hub Secrets

1. **Create Docker Hub Access Token:**
   - Go to [Docker Hub Security Settings](https://hub.docker.com/settings/security)
   - Click "New Access Token"
   - Name: `GitHub Actions CI/CD`
   - Permissions: `Read, Write, Delete`
   - Copy the token (shown only once!)

2. **Add Secrets to GitHub:**
   - Go to your repository on GitHub
   - Navigate to: `Settings` → `Secrets and variables` → `Actions`
   - Click "New repository secret"
   - Add these two secrets:

   **Secret 1:**
   - Name: `DOCKER_HUB_USERNAME`
   - Value: Your Docker Hub username (e.g., `jurel89`)

   **Secret 2:**
   - Name: `DOCKER_HUB_TOKEN`
   - Value: The access token you just created

## Step 2: Verify Workflows

All workflow files are already in place:

```
.github/workflows/
├── pr-validation.yml      # Validates PRs to main
├── main-build.yml         # Builds images on main
├── release.yml            # Publishes releases to Docker Hub
└── security-quality.yml   # Quick checks on feature branches
```

## Step 3: Test the Pipeline

### Test 1: Feature Branch Push (Optional)

```bash
# Create a test branch
git checkout -b test/ci-pipeline

# Make a small change
echo "# CI/CD Test" >> README.md
git add README.md
git commit -m "test: CI/CD pipeline"
git push origin test/ci-pipeline

# Check GitHub Actions tab
# → security-quality.yml should run
```

### Test 2: Pull Request to Main

```bash
# Create PR from your current branch to main on GitHub
# → pr-validation.yml will run automatically
# → All validation checks should pass
# → Review the PR summary
```

### Test 3: Merge to Main

```bash
# Merge the PR on GitHub
# → main-build.yml runs automatically
# → Docker image builds successfully
# → Artifact stored for 7 days
```

### Test 4: Create Release

```bash
# Checkout main and pull latest
git checkout main
git pull origin main

# Create version tag
git tag -a v0.1.0 -m "Release v0.1.0: Initial CI/CD setup"
git push origin v0.1.0

# Check GitHub Actions tab
# → release.yml runs
# → Image pushed to Docker Hub
# → GitHub release created
```

## Step 4: Verify Docker Hub

1. Go to [Docker Hub](https://hub.docker.com/)
2. Navigate to your repository (e.g., `jurel89/homelab-subtitle-service`)
3. Verify tags are present:
   - `0.1.0`
   - `0.1`
   - `0`
   - `latest`

## Step 5: Test Docker Image

```bash
# Pull the published image
docker pull <your-username>/homelab-subtitle-service:latest

# Test it works
docker run --rm <your-username>/homelab-subtitle-service:latest subsvc --help
```

## Optional: Self-Hosted Runner

If you want automatic deployment to your homelab:

1. **Install Runner on Homelab Server:**
   ```bash
   # Follow instructions in docs/CI_CD_WORKFLOWS.md
   # Section: "Self-Hosted Runner Setup"
   ```

2. **Uncomment Deployment Job:**
   - Edit `.github/workflows/main-build.yml`
   - Uncomment the `deploy-selfhosted` job
   - Update volume paths for your environment

3. **Test Deployment:**
   ```bash
   # Push to main
   git push origin main
   
   # Check your homelab server
   docker ps | grep homelab-subtitle-service
   ```

## Troubleshooting

### Workflow fails with "Error: Unable to locate credentials"

**Solution:** Verify Docker Hub secrets are correctly set:
- `DOCKER_HUB_USERNAME` (your Docker Hub username)
- `DOCKER_HUB_TOKEN` (your access token, not password)

### Release workflow doesn't trigger

**Solution:** Ensure you're pushing tags with `v` prefix:
```bash
# ✅ Correct
git tag -a v1.0.0 -m "Release v1.0.0"

# ❌ Wrong
git tag -a 1.0.0 -m "Release 1.0.0"
```

### Docker build fails on Hadolint

**Solution:** Review Dockerfile issues reported by Hadolint:
1. Check workflow logs
2. Download `hadolint-report.sarif` artifact
3. Fix Dockerfile according to best practices

### Tests fail on PR

**Solution:** Run tests locally before pushing:
```bash
# Run tests
.venv/bin/python -m pytest -m "not integration"

# Run linting
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

## Workflow Behavior Summary

| Event | Workflow | Actions |
|-------|----------|---------|
| Push to feature branch | `security-quality.yml` | Quick validation |
| PR to main | `pr-validation.yml` | Full validation, blocks merge |
| Merge to main | `main-build.yml` | Build image, store artifact |
| Push version tag | `release.yml` | Publish to Docker Hub, create release |

## Next Steps

Once setup is complete:

1. ✅ All workflows working
2. ✅ Docker Hub receiving images
3. ✅ Releases created automatically

Continue with development:
- See `docs/CI_CD_WORKFLOWS.md` for detailed workflow documentation
- See `pendings.md` for upcoming features
- Follow the standard Git workflow: feature branch → PR → main → tag → release

## Support

If you encounter issues:
1. Check workflow logs in GitHub Actions tab
2. Review `docs/CI_CD_WORKFLOWS.md`
3. Open an issue on GitHub

---

**You're all set!** 🚀 The CI/CD pipeline is ready to automate your releases.
