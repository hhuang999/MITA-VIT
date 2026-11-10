"""Source provenance in the exported course code."""

import pytest

from fastweight_vit.io import source_commit


def test_export_records_the_source_revision_without_git(tmp_path):
    revision = 'b3ae06df55b264e3e32b40c9c132923fa117601c'
    (tmp_path / '.git_archival.txt').write_text(revision + '\n', encoding='utf-8')
    assert source_commit(tmp_path) == revision


@pytest.mark.parametrize('marker', [None, '$Format:%H$'])
def test_export_requires_an_expanded_source_revision(tmp_path, marker):
    if marker is not None:
        (tmp_path / '.git_archival.txt').write_text(marker, encoding='utf-8')
    with pytest.raises(RuntimeError, match='source revision'):
        source_commit(tmp_path)
