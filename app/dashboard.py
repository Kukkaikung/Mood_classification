"""
AudiMood — offline analytics dashboard.
Reads the session CSVs written by app/inference.py and shows how the audience's
emotions changed over the session.

Run:  streamlit run app/dashboard.py
"""
import os
import glob
import pandas as pd
import plotly.express as px
import streamlit as st


EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# same hues as the mood panel in inference.py, darkened so they read on a white background
EMOTION_COLORS = {
    'angry'   : '#d62728',
    'disgust' : '#ff7f0e',
    'fear'    : '#9467bd',
    'happy'   : '#2ca02c',
    'neutral' : '#17becf',
    'sad'     : '#1f77b4',
    'surprise': '#bcbd22',
}

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR     = os.path.join(PROJECT_DIR, 'logs')


def list_sessions(log_dir):
    # newest first
    return sorted(glob.glob(os.path.join(log_dir, 'session_*.csv')), reverse=True)


def load_session(source):
    df = pd.read_csv(source)
    missing = [c for c in ['elapsed_s', 'total_faces'] + EMOTIONS if c not in df.columns]
    if missing:
        raise ValueError(f'not an AudiMood session file — missing columns: {missing}')
    return df


def summarise_segments(df, segment_s):
    # group rows into fixed time windows and compute each emotion's share of confident faces
    df = df.copy()
    df['segment'] = (df['elapsed_s'] // segment_s).astype(int)
    seg = df.groupby('segment').agg(
        **{e: (e, 'sum') for e in EMOTIONS},
        avg_faces=('total_faces', 'mean'),
    )
    confident = seg[EMOTIONS].sum(axis=1)
    shares    = seg[EMOTIONS].div(confident.where(confident > 0), axis=0)   # NaN if no confident faces

    seg['start_s']  = seg.index * segment_s
    seg['dominant'] = seg[EMOTIONS].idxmax(axis=1).where(confident > 0, 'none')
    seg['dominant_share'] = shares.max(axis=1)
    return seg, shares


def fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f'{m:02d}:{s:02d}'


def main():
    st.set_page_config(page_title='AudiMood Dashboard', layout='wide')
    st.title('AudiMood — Audience Emotion Dashboard')
    st.caption('Retrospective analysis of session logs recorded by the real-time detector (app/inference.py)')

    # sidebar: pick a session and the analysis window
    st.sidebar.header('Session')
    sessions = list_sessions(LOG_DIR)
    uploaded = st.sidebar.file_uploader('...or upload a session CSV', type='csv')

    if uploaded is not None:
        source, name = uploaded, uploaded.name
    elif sessions:
        source = st.sidebar.selectbox('Session file', sessions, format_func=os.path.basename)
        name   = os.path.basename(source)
    else:
        st.info(f'No session logs found in `{LOG_DIR}`. Run `python app/inference.py` first, '
                'or upload a CSV in the sidebar.')
        return

    segment_s = st.sidebar.slider('Segment length (seconds)', 5, 120, 10, step=5)

    try:
        df = load_session(source)
    except Exception as err:
        st.error(f'Could not read {name}: {err}')
        return
    if df.empty:
        st.warning(f'{name} has no rows yet.')
        return

    seg, shares = summarise_segments(df, segment_s)
    totals      = df[EMOTIONS].sum()
    n_confident = int(totals.sum())

    # 1) headline numbers
    st.subheader(name)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Duration', fmt_time(df['elapsed_s'].max()))
    c2.metric('Average faces', f"{df['total_faces'].mean():.1f}")
    c3.metric('Dominant emotion', totals.idxmax() if n_confident else '—')
    if 'uncertain' in df.columns and df['total_faces'].sum() > 0:
        c4.metric('Uncertain detections', f"{df['uncertain'].sum() / df['total_faces'].sum():.0%}")

    if n_confident == 0:
        st.warning('No confident emotion predictions in this session.')
        return

    # 2) emotion trend over time
    st.markdown(f'#### Emotion trend (share of audience, {segment_s}s segments)')
    trend = shares.assign(time=seg['start_s'].map(fmt_time)).melt(
        id_vars='time', value_vars=EMOTIONS, var_name='emotion', value_name='share')
    fig = px.line(trend, x='time', y='share', color='emotion', markers=True,
                  color_discrete_map=EMOTION_COLORS, category_orders={'emotion': EMOTIONS})
    fig.update_layout(yaxis_tickformat='.0%', yaxis_range=[0, 1], xaxis_title='Time (mm:ss)',
                      yaxis_title='Share of faces', legend_title='')
    st.plotly_chart(fig, width='stretch')

    left, right = st.columns([1, 1])

    # 3) overall share for the whole session
    with left:
        st.markdown('#### Overall emotion share')
        overall = pd.DataFrame({'emotion': EMOTIONS, 'share': (totals / n_confident).values})
        fig = px.bar(overall, x='emotion', y='share', color='emotion',
                     color_discrete_map=EMOTION_COLORS, category_orders={'emotion': EMOTIONS})
        fig.update_layout(yaxis_tickformat='.0%', showlegend=False, xaxis_title='',
                          yaxis_title='Share of faces')
        st.plotly_chart(fig, width='stretch')

    # 4) dominant emotion per segment
    with right:
        st.markdown('#### Dominant emotion per segment')
        table = pd.DataFrame({
            'Segment'  : [f'{fmt_time(s)}–{fmt_time(s + segment_s)}' for s in seg['start_s']],
            'Avg faces': seg['avg_faces'].round(1),
            'Dominant' : seg['dominant'],
            'Share'    : seg['dominant_share'].map(lambda v: f'{v:.0%}' if pd.notna(v) else '—'),
        })
        st.dataframe(table, hide_index=True, width='stretch')

    with st.expander('Raw session data'):
        st.dataframe(df, hide_index=True, width='stretch')


main()
