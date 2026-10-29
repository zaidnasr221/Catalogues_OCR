"""Offline checks for the independently executed compact Markdown reader."""
import json
from pathlib import Path
from PIL import Image
import pytest
from as_ocr import page_markdown as md
from as_ocr.qwen import ModelConfig,ModelError

pytestmark=pytest.mark.usefixtures('portable_checkpoint_io')


class Client:
    def __init__(self,*responses):
        self.responses=iter(responses);self.requests=0;self.contexts=[]
        self.config=ModelConfig(timeout=180,max_requests=16,max_tokens=1536,retries=0)
    def request_json(self,image,prompt,context,**kw):
        assert max(image.size)<=2048
        self.requests+=1;self.contexts.append(json.loads(context));result=next(self.responses)
        if isinstance(result,Exception):raise result
        return json.dumps(result)


def reading(value,complete=True,unreadable=()):return dict(markdown=value,complete=complete,unreadable=list(unreadable))
def audit(complete=True,issues=()):return dict(complete=complete,missing_regions=[],issues=list(issues))
def layout(*regions):return dict(coverage='complete',regions=[dict(kind=kind,box=box,rotation=0) for kind,box in regions],uncertainty=[])


def run(tmp_path,monkeypatch,client):
    def render(source,page,geometry,box,maximum):
        assert maximum<=2048
        return Image.new('RGB',(100,100),'white'),dict(pixel_size=[100,100],source_to_pixels=[1,0,0,1,0,0],pixels_to_source=[1,0,0,1,0,0])
    monkeypatch.setattr(md,'render_text_region',render)
    return md.extract_page_compact(Path('not-opened.pdf'),dict(id='p',document_id='doc',page_index=6),
        dict(source_rect=[0,0,100,100]),[],tmp_path/'evidence',client)


def test_compact_schema_does_not_repeat_text_as_cells_or_per_label_boxes():
    assert set(md.TextReading.model_fields)=={'markdown','complete','unreadable'}
    assert 'NEVER pixels or 0-1000' in md.LAYOUT_PROMPT
    assert all(p['minimum']==0 and p['maximum']==1 for p in md.TextRegion.model_json_schema()['properties']['box']['prefixItems'])


def test_table_headers_values_blank_cells_and_duplicate_rows_preserved():
    value='| Code | Width | Note |\n|---|---|---|\n| 001.20 - A | 22 | |\n| 001.20 - A | 22 | |'
    table=md.markdown_tables(value)[0]
    assert table['headers']==['Code','Width','Note'] and table['valid']
    assert table['rows']==[['001.20 - A','22',''],['001.20 - A','22','']]


def test_continuation_merge_removes_only_same_original_header():
    one='| Code | Value |\n|---|---|\n| 001 | 6 |\n| 001 | 6 |'
    merged,ok=md._merge_table_parts(one,one)
    assert ok and len(md.markdown_tables(merged)[0]['rows'])==4
    assert merged.count('| Code | Value |')==1


def test_drawings_are_text_bullets_without_images():
    value,issues=md.clean_transcription('32496\nD/G spacer holder\n6\n6\n![drawing](x.png)','drawing')
    assert value.count('- 6')==2 and '- 32496' in value and '![' not in value and issues


def test_crop_limit_matches_real_transport_limit():
    md.CompactConfig().validate()
    with pytest.raises(ValueError):md.CompactConfig(crop_max=4096).validate()


def test_readable_regions_survive_other_region_timeout(tmp_path,monkeypatch):
    client=Client(layout(('prose',[0,0,.5,1]),('drawing',[.5,0,1,1])),
        ModelError('Qwen request failed: TimeoutError'),reading('Readable original paragraph'),
        ModelError('Qwen request failed: TimeoutError'),audit(False,['Lower prose area is unreadable']),
        reading('32496\n22 mm'),audit(),audit(False,['Lower prose area is unreadable']))
    result=run(tmp_path,monkeypatch,client);text=(tmp_path/'evidence/page.md').read_text()
    assert result['status']=='PARTIAL' and 'Readable original paragraph' in text and '- 32496' in text
    assert len(result['failures'])==2 and client.requests==8


def test_table_has_independent_source_verification(tmp_path,monkeypatch):
    value='| Code | Width |\n|---|---|\n| 00120 | 22 |'
    client=Client(layout(('table',[0,0,1,1])),reading(value),audit(),audit())
    result=run(tmp_path,monkeypatch,client)
    assert result['status']=='COMPLETE' and result['units'][0]['table_audit']['complete']
    assert client.contexts[2]['table_structure'][0]['body_rows']==1


def test_failed_table_check_does_not_erase_readable_cells(tmp_path,monkeypatch):
    value='| Code | Width |\n|---|---|\n| 00120 | [UNREADABLE] |'
    client=Client(layout(('table',[0,0,1,1])),reading(value,False,['Width unreadable']),audit(False,['Width unverified']),audit(False,['Width unverified']))
    result=run(tmp_path,monkeypatch,client)
    assert result['status']=='PARTIAL' and '00120' in (tmp_path/'evidence/page.md').read_text()


def test_layout_failure_runs_full_width_recovery_instead_of_quadrants(tmp_path,monkeypatch):
    client=Client(ModelError('Layout timeout'),reading('A'),audit(),reading('B'),audit(),reading('C'),audit(),audit())
    result=run(tmp_path,monkeypatch,client)
    panels=[r['context']['source_panels'][0]['source_box'] for r in result['requests'] if r['id'].startswith('region-')]
    assert all(b[0]==0 and b[2]==100 for b in panels)
    assert result['status']=='PARTIAL'


def test_native_supplement_counts_repeated_physical_text_locations():
    spans=[(i,dict(text='6',bbox=[i*10,0,i*10+5,5])) for i in range(4)]
    assert md.native_supplements(spans,'- 6\n- 6')==['6','6']
    assert md.native_supplements(spans,'32496')==['6']*4


def test_headerless_html_table_keeps_cells_without_inventing_header():
    tables=md.markdown_tables('<table><tr><td>00120</td><td>22</td></tr><tr><td>00120</td><td></td></tr></table>')
    assert len(tables)==1 and tables[0]['headers']==[]
    assert tables[0]['rows']==[['00120','22'],['00120','']]


def test_rotated_source_text_is_made_upright_without_exceeding_pixel_cap():
    image=Image.new('RGB',(200,700),'white')
    mapping=dict(source_to_pixels=[1,0,0,1,0,0],pixels_to_source=[1,0,0,1,0,0],pixel_size=[200,700])
    rotated,new=md.upright(image,mapping,90)
    assert rotated.size==(700,200) and new['pixel_size']==[700,200]


def test_overlapping_repair_replaces_same_source_scope_instead_of_duplicate_text(tmp_path,monkeypatch):
    repair=dict(complete=False,missing_regions=[dict(kind='other',box=[0,0,1,1],rotation=0)],issues=['Missing one label'])
    client=Client(layout(('prose',[0,0,1,1])),reading('CERT\nISO 9001'),audit(),repair,reading('CERT\nISO 9001\nQUALICOAT'))
    result=run(tmp_path,monkeypatch,client);text=(tmp_path/'evidence/page.md').read_text()
    assert text.count('ISO 9001')==1 and 'QUALICOAT' in text
    assert result['status']=='PARTIAL'


def test_source_rejection_masks_exact_annotation_without_changing_other_ids(tmp_path,monkeypatch):
    check=dict(complete=False,missing_regions=[],issues=['Dimension uncertain'],unsupported_text=['6'])
    client=Client(layout(('drawing',[0,0,1,1])),reading('32496\n6'),check,audit(False,['Uncertain dimension']))
    result=run(tmp_path,monkeypatch,client);text=(tmp_path/'evidence/page.md').read_text()
    assert '- 32496' in text and '- [UNREADABLE]' in text and result['status']=='PARTIAL'


def test_read_and_source_audit_reuse_one_region_render(tmp_path,monkeypatch):
    client=Client(layout(('drawing',[0,0,1,1])),reading('32496\n6\n6'),audit(),audit())
    calls=[]
    def render(source,page,geometry,box,maximum):
        calls.append((tuple(box),maximum))
        return Image.new('RGB',(100,100),'white'),dict(pixel_size=[100,100],source_to_pixels=[1,0,0,1,0,0],pixels_to_source=[1,0,0,1,0,0])
    monkeypatch.setattr(md,'render_text_region',render)
    result=md.extract_page_compact(Path('not-opened.pdf'),dict(id='p',document_id='doc',page_index=6),
        dict(source_rect=[0,0,100,100]),[],tmp_path/'evidence',client)
    assert len(calls)==2  # overview and region; independent audit still executes
    assert client.requests==4 and result['status']=='COMPLETE'
    assert result['units'][0]['markdown'].count('- 6')==2


@pytest.mark.parametrize('native_accepted',[True,False])
def test_native_prose_saves_only_visually_verified_transcription(tmp_path,monkeypatch,native_accepted):
    responses=[layout(('prose',[0,0,1,1])),audit(native_accepted,[] if native_accepted else ['Missing outlined annotation'])]
    if not native_accepted:responses += [reading('Original prose\n32496'),audit()]
    responses.append(audit())
    client=Client(*responses)
    def render(*args):return Image.new('RGB',(100,100),'white'),dict(pixel_size=[100,100],source_to_pixels=[1,0,0,1,0,0],pixels_to_source=[1,0,0,1,0,0])
    monkeypatch.setattr(md,'render_text_region',render)
    result=md.extract_page_compact(Path('not-opened.pdf'),dict(id='p',document_id='doc',page_index=6),
        dict(source_rect=[0,0,100,100]),[(0,dict(text='Original prose',bbox=[10,10,50,20]))],
        tmp_path/'evidence',client,md.CompactConfig(prefer_native=True))
    assert result['status']=='COMPLETE' and client.requests==(3 if native_accepted else 5)
    assert bool(result['units'][0].get('native_verified'))==native_accepted
    if not native_accepted:assert '32496' in (tmp_path/'evidence/page.md').read_text()


def test_native_failed_visual_checks_keep_readable_partial_text(tmp_path,monkeypatch):
    client=Client(layout(('prose',[0,0,1,1])),ModelError('native audit timeout'),
        ModelError('read timeout'),ModelError('read timeout'),ModelError('read timeout'),
        audit(False,['Unverified source prose']),audit(False,['Unverified source prose']))
    def render(*args):return Image.new('RGB',(100,100),'white'),dict(pixel_size=[100,100],source_to_pixels=[1,0,0,1,0,0],pixels_to_source=[1,0,0,1,0,0])
    monkeypatch.setattr(md,'render_text_region',render)
    result=md.extract_page_compact(Path('not-opened.pdf'),dict(id='p',document_id='doc',page_index=6),
        dict(source_rect=[0,0,100,100]),[(0,dict(text='Readable source',bbox=[10,10,50,20]))],
        tmp_path/'evidence',client,md.CompactConfig(prefer_native=True,max_requests=3))
    assert result['status']=='PARTIAL' and 'Readable source' in (tmp_path/'evidence/page.md').read_text()
