# CI Workflow Migration Notes

## Dynamic Platform Discovery

The CI now uses **dynamic platform discovery** via `discover-platforms` job in `.github/workflows/ci.yml`:

```yaml
discover-platforms:
  runs-on: ubuntu-latest
  outputs:
    platforms: ${{ steps.detect.outputs.platforms }}
  steps:
    - name: Detect configured platforms
      run: |
        platforms=$(find .github/configs -maxdepth 1 -type f -name '*.yml' \
          -exec basename {} .yml \; | sort | jq -R -s -c 'split("\n")[:-1]')
        echo "platforms=$platforms" >> "$GITHUB_OUTPUT"

platform-pipeline:
  needs: discover-platforms
  strategy:
    matrix:
      platform: ${{ fromJson(needs.discover-platforms.outputs.platforms) }}
  uses: ./.github/workflows/all-tests-common.yml
  with:
    platform: ${{ matrix.platform }}
```

## Impact on Required Status Checks

### Before Migration
- Static job names like `build-wheel-cuda`, `integration-test-cuda`, etc.
- Branch protection rules referenced these exact names
- Adding a new platform required updating branch protection manually

### After Migration
- Dynamic job names: `Platform pipeline (cuda)`, `Platform pipeline (musa)`, etc.
- Job names are parameterized by the matrix platform value
- Adding a new platform (e.g., creating `.github/configs/newplatform.yml`) automatically creates `Platform pipeline (newplatform)` job

### Required Actions for Repository Administrators

**If branch protection is enabled on `main` or other protected branches:**

1. Check current required status checks:
   ```bash
   gh api repos/{owner}/{repo}/branches/main/protection | jq '.required_status_checks.contexts'
   ```

2. Update required checks to use the new dynamic job names:
   - **Remove old static names:** `build-wheel-cuda`, `integration-test-cuda`, etc.
   - **Add new dynamic names:** `Platform pipeline (cuda)`, `Platform pipeline (musa)`, etc.

3. If using GitHub UI:
   - Navigate to: Settings → Branches → Branch protection rules → Edit rule for `main`
   - Under "Status checks that are required", remove old job names
   - After next CI run, the new job names will appear in the search box
   - Select all `Platform pipeline (*)` jobs

4. If using GitHub API:
   ```bash
   gh api -X PUT repos/{owner}/{repo}/branches/main/protection \
     -f required_status_checks[contexts][]=lint \
     -f required_status_checks[contexts][]="Platform pipeline (cuda)" \
     -f required_status_checks[contexts][]="Platform pipeline (musa)" \
     # ... add all other platforms
   ```

### External Caller Impact

**If other workflows call the deleted reusable workflows:**

The following reusable workflows were **not deleted** but are now redundant:
- `.github/workflows/build-wheel-*.yml`
- `.github/workflows/integration-test-*.yml`

These still exist for backward compatibility but are no longer called by `ci.yml`.

**For external repositories or workflows that reference them:**
- **Old style (still works but deprecated):**
  ```yaml
  uses: BrianPei/PyTorch-Plugin-FL/.github/workflows/build-wheel-cuda.yml@main
  ```

- **New style (recommended):**
  ```yaml
  uses: BrianPei/PyTorch-Plugin-FL/.github/workflows/all-tests-common.yml@main
  with:
    platform: cuda
    run_build: true
    run_integration_tests: true
  ```

### Verification Commands

After migration, verify the CI is working correctly:

```bash
# Check that all platform configs are discovered
python -c "
import json
from pathlib import Path
platforms = sorted(p.stem for p in Path('.github/configs').glob('*.yml'))
print('Discovered platforms:', json.dumps(platforms))
"

# Validate all manifests
python .github/scripts/validate_integration_manifests.py --configs-dir .github/configs

# Check workflow syntax
find .github/workflows -name "*.yml" -exec echo "Checking {}" \; -exec actionlint {} \;
```

## Rollback Instructions

If issues are discovered post-migration:

1. **Restore static per-platform workflows:**
   ```bash
   git revert <migration-commit-sha>
   ```

2. **Update branch protection to use old job names**

3. **Investigate the root cause** before attempting migration again

## Summary

- ✅ **Dynamic discovery** reduces maintenance burden
- ✅ **New platforms** automatically get CI coverage
- ⚠️ **Branch protection rules** must be updated manually
- ⚠️ **External callers** should migrate to `all-tests-common.yml`
- ℹ️ **Old workflows** remain for compatibility but are not actively used
