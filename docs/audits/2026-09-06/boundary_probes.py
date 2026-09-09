"""Small independent probes; only synthetic temporary files and in-memory data."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile

from core.langgraph.content_manager import ContentManager
from core.langgraph.export import _extract_body
from core.project_config import NarrativeProjectConfig
from core.project_manager import ProjectManager

results = {}
with tempfile.TemporaryDirectory(prefix='saga-content-probe-') as temp:
    root = Path(temp)
    project = root / 'project'
    manager = ContentManager(str(project))
    # Same-prefix sibling escapes actual content root.
    escaped = manager.save_text('synthetic-only', '../content_sibling', 'sample')
    results['content_bucket_sibling_escape'] = {
        'outside_content_root': not (project / escaped['path']).resolve().is_relative_to(manager.content_dir.resolve()),
        'returned_path': escaped['path'],
    }
    outside = root / 'synthetic.txt'
    outside.write_text('synthetic outside payload')
    data = outside.read_bytes()
    ref = {'path': '../synthetic.txt', 'size_bytes': len(data), 'checksum': hashlib.sha256(data).hexdigest()}
    results['strict_read_outside_project'] = manager.load_text_strict(ref) == 'synthetic outside payload'
    manager.delete(ref)
    results['delete_outside_project'] = not outside.exists()
    old = manager.save_text('version-one-original', 'drafts', 1, 1)
    manager.save_text('version-one-overwritten', 'drafts', 1, 1)
    try:
        manager.load_text_strict(old)
    except ValueError as exc:
        results['same_version_invalidates_prior_ref'] = type(exc).__name__
    except Exception as exc:
        results['same_version_invalidates_prior_ref'] = type(exc).__name__
    immutable = manager.save_text('immutable probe', 'drafts', 2, 1)
    immutable |= {'path': 'changed-by-inplace-union'}
    results['frozen_ref_inplace_union_mutates'] = immutable['path'] == 'changed-by-inplace-union'
    ProjectManager.projects_root = root / 'stories'
    first = NarrativeProjectConfig(title='Same Title!', genre='fiction', theme='first', setting='synthetic', protagonist_name='A', narrative_style='third-person limited', total_chapters=1)
    second = first.model_copy(update={'title':'Same Title?', 'theme':'second'})
    location = ProjectManager.save_config(first, review=False)
    ProjectManager.save_config(second, review=False)
    results['sanitized_title_collision_overwrites_config'] = ProjectManager.load_config(location).theme == 'second'
    chapters = location / 'chapters'
    chapters.mkdir()
    (chapters / 'chapter_notes.md').write_text('not a chapter')
    results['noncanonical_file_counted_as_complete'] = ProjectManager.count_completed_chapters(location)

prose = '---A dialogue dash, not front matter\nThis prose should survive.\n---\nEnding.'
results['export_non_frontmatter_prose_removed'] = _extract_body(prose) == 'Ending.'
# Execute only the CLI parser with all application callbacks absent; --help above
# already demonstrated the real imported CLI can run in the isolated environment.
print(json.dumps(results, indent=2))
