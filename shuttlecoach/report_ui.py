"""Visible AI findings with local measurement and video evidence."""
import json
import os
from pathlib import Path

import streamlit as st

from . import coaching, credentials, llm_report


def fact_caption(fact):
    m=fact['metrics']
    if fact['kind'] in {'selected_window_overview','positioning_time_window'}:
        return (f"Source {fact['start_s']:.1f}–{fact['end_s']:.1f}s · tracked {m['tracked_time']} · missing {m['missing_time']} · "
                f"accepted court positions: front {m['front']}, middle {m['middle']}, rear {m['rear']}. Missing positions are excluded from these shares.")
    if fact['kind']=='unconfirmed_footwork_candidates':
        return (f"Usable poses: {m['pose_coverage']} · lunge-shape candidates {m['lunge']} · stance-opening candidates {m['stance_opening']} · "
                f"left/right foot-movement candidates {m['left_step']}/{m['right_step']}. These are not confirmed technique events.")
    if 'confirmed_events' in m:return f"Confirmed timing events: {m['confirmed_events']} · median interval {m['median_interval']}."
    return 'Based on the locally approved pattern evidence.'


def show_ai_analysis(run, manifest, active=False):
    run=Path(run)
    st.subheader('AI analysis')
    st.write('Exact measurements, what they support, and what to check on video. Comparisons require sufficiently covered time windows; practice still requires validated evidence.')
    try:saved_key=credentials.load_key()
    except (ValueError,OSError) as exc:
        saved_key=None
        st.warning(str(exc))
    with st.expander('AI connection settings'):
        st.link_button('Get a free Groq API key','https://console.groq.com/keys')
        st.caption('Anonymized measurements go to Groq; video, names and local paths stay on this Mac. Free-plan quotas apply. No automatic retries or paid upgrades.')
        if saved_key:
            if credentials.can_save_key():st.success('A Groq API key is saved in macOS Keychain and will be used automatically. Enter a key below only to replace it.')
            else:st.success('A Groq API key is available from the Docker runtime secret and will be used automatically.')
        key=st.text_input('Groq API key',type='password',key='groq_api_key')
        remember=False
        if credentials.can_save_key():
            remember=st.checkbox('Remember API key on this Mac',value=True,key='remember_groq_key')
            if st.button('Save API key on this Mac',disabled=not key.strip()):
                try:
                    credentials.save_key(key)
                    st.rerun()
                except (ValueError,OSError) as exc:st.error(str(exc))
        else:st.caption('Container keys come from the runtime secret; a key entered here lasts for this browser session. Restart with make docker-up to reuse the key saved on your Mac.')
    if st.button('Analyze tracking and footwork',disabled=active or not (key.strip() or os.environ.get('GROQ_API_KEY') or saved_key),type='primary'):
        try:
            if remember and key.strip():credentials.save_key(key)
            with st.spinner('Analyzing positioning and footwork measurements…'):llm_report.generate_llm_report(run,key or saved_key)
            st.rerun()
        except (ValueError,OSError) as exc:st.error(str(exc))
    path=run/'llm_report.json'
    if not path.exists():
        st.info('Run AI analysis to see player-by-player findings here. No download is needed.')
        return
    draft=json.loads(path.read_text())
    if draft.get('report_version')!=llm_report.REPORT_VERSION:
        st.info('A stricter report format is available. Click Analyze tracking and footwork to replace the previous draft with exact measurements, interpretation and specific video checks.')
        return
    try:current=draft.get('evidence_signature')==coaching.evidence_signature(run)
    except (OSError,ValueError,KeyError):current=False
    if not current:
        st.info('Tracking or review evidence changed. Run AI analysis to update the findings.')
        return
    st.write(draft['summary'])
    if draft['mode']!='coaching':st.caption('Descriptive analysis: measured patterns can be discussed, but unvalidated tracking and unconfirmed footwork do not support technique diagnoses or personalized drills.')
    facts={r['id']:r for r in draft['facts']}
    for player in draft['analyses']:
        with st.container(border=True):
            st.subheader(draft['player_labels'][player['player_id']])
            overview=next((r for r in draft['facts'] if r['player_id']==player['player_id'] and r['kind']=='selected_window_overview'),None)
            if overview:
                m=overview['metrics'];a,b,c=st.columns(3)
                a.metric('Tracking available',m['tracking_coverage'])
                zones=[z for z in ['front','middle','rear'] if m[z]!='unavailable']
                dominant=max(zones,key=lambda z:float(m[z].rstrip('%'))) if zones else None
                b.metric('Most observed zone',f"{dominant.title()} · {m[dominant]}" if dominant else 'Unavailable')
                c.metric('No accepted position',m['missing_time'])
                st.caption('Zone shares use accepted in-court time; they exclude missing positions and may include pauses.')
            if 'change' not in llm_report.player_requirements(draft,player['player_id'])['required_topics']:
                st.caption('Time comparison unavailable: it needs two non-overlapping source windows, each with at least 95% tracking availability and 5 accepted in-court seconds. A whole-window percentage does not establish a change over time.')
            for number,finding in enumerate(player['observations'],1):
                titles={'positioning':'Positioning & tracking reliability','change':'Time-window comparison','footwork':'Footwork evidence'}
                st.markdown(f"**{titles[finding['topic']]}**")
                st.write(finding['finding'])
                st.markdown('**What this means**')
                st.write(finding['meaning'])
                st.markdown('**What to review**')
                st.write(finding['review_focus'])
                with st.expander(f'Measurements and video · finding {number}'):
                    sources=[facts[ref] for ref in finding['evidence_ids']]
                    for fact in sources:st.caption(fact_caption(fact))
                    timed=[r for r in sources if r['start_s'] is not None]
                    if timed:
                        selected=st.selectbox('Source window',range(len(timed)),format_func=lambda i:f"{timed[i]['start_s']:.1f}–{timed[i]['end_s']:.1f}s",
                                              key=f"ai_window_{run.name}_{player['player_id']}_{number}") if len(timed)>1 else 0
                        original=manifest['video'].get('original_source',manifest['video'])
                        st.video(original['path'],start_time=max(0,int(timed[selected]['start_s'])-1))
            st.caption(player['limitations'])
    if draft['sections']:
        local_report=json.loads((run/'coaching.json').read_text())
        if local_report['gate']=='passed' and local_report.get('evidence_signature')==draft['evidence_signature']:
            st.markdown('**Practice supported by the reviewed evidence**')
            for section in draft['sections']:
                item=local_report['suggestions'][section['id']-1]
                with st.container(border=True):
                    st.markdown(f"**{item['player']} · {item.get('title','Positioning practice')}**")
                    st.write(section['explanation']);st.write(section['practice'])
    st.caption(draft['limitations'])
