# Release Process

This document describes the release process for ezpz.

## Versioning Scheme

ezpz follows [Semantic Versioning](https://semver.org/) (SemVer):

- MAJOR version when you make incompatible API changes
- MINOR version when you add functionality in a backwards compatible manner
- PATCH version when you make backwards compatible bug fixes

## Release Steps

### 1. Preparation

1. **Create a release branch**:
   ```bash
   git checkout -b release/vX.Y.Z
   ```

2. **Update the version** in `src/ezpz/__about__.py`:
   ```python
   __version__ = "X.Y.Z"
   ```

3. **Update CHANGELOG.md**:
   - Add a new section for the release
   - Include all notable changes
   - Link to relevant issues and pull requests

4. **Run tests**:
   ```bash
   hatch run test
   ```

5. **Run linting**:
   ```bash
   hatch run lint
   ```

6. **Build documentation**:
   ```bash
   mkdocs build
   ```

### 2. Create Release

1. **Commit changes**:
   ```bash
   git add .
   git commit -m "Release vX.Y.Z"
   ```

2. **Create and push tag**:
   ```bash
   git tag -a vX.Y.Z -m "Release vX.Y.Z"
   git push origin vX.Y.Z
   ```

3. **Push release branch**:
   ```bash
   git push origin release/vX.Y.Z
   ```

### 3. Create GitHub Release

1. Go to the [GitHub Releases page](https://github.com/saforem2/ezpz/releases)
2. Click "Draft a new release"
3. Select the tag you just created
4. Set the release title to "vX.Y.Z"
5. Copy the release notes from CHANGELOG.md
6. Click "Publish release"

### 4. Post-Release

1. **Merge release branch** to main:
   ```bash
   git checkout main
   git merge release/vX.Y.Z
   git push origin main
   ```

2. **Update development branch**:
   ```bash
   git checkout dev
   git merge main
   git push origin dev
   ```

3. **Delete release branch**:
   ```bash
   git branch -d release/vX.Y.Z
   git push origin --delete release/vX.Y.Z
   ```

## Automated Release Process

For future releases, we plan to implement an automated release process using GitHub Actions that will:

1. Automatically bump version numbers
2. Generate release notes
3. Create GitHub releases
4. Publish to PyPI

## Hotfix Releases

For critical bug fixes, create a hotfix branch from the latest release tag:

```bash
git checkout -b hotfix/vX.Y.Z
# Make fixes
git commit -am "Fix critical issue"
git tag -a vX.Y.Z -m "Hotfix vX.Y.Z"
git push origin vX.Y.Z
```

Then follow the same release process as above.