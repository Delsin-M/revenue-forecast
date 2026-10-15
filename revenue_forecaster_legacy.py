import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title='Revenue Forecasting Simulator', page_icon='📈', layout='wide')

REQUIRED = ['Program', 'Status', 'Amount', 'Start Date', 'End Date', 'Win Probability', 'Funding Agency']
DEFAULT = pd.DataFrame([
    ['Booked Program Alpha','Booked',1200000,'2026-01-01','2026-12-31',1.00,'DARPA'],
    ['Bid Program Beta','Bid',900000,'2026-04-01','2027-03-31',0.65,'AFRL'],
    ['Bid Program Gamma','Bid',600000,'2026-07-01','2027-06-30',0.35,'ONR'],
], columns=REQUIRED)
DEFAULT["Start Date"] = pd.to_datetime(DEFAULT["Start Date"])
DEFAULT["End Date"] = pd.to_datetime(DEFAULT["End Date"])

st.title('Revenue Forecasting Simulator')
st.caption('Linear monthly spend allocation with Monte Carlo simulation of bid wins and losses.')

with st.sidebar:
    st.header('Simulation settings')
    runs = st.number_input('Monte Carlo runs', min_value=100, max_value=250000, value=10000, step=1000)
    seed = st.number_input('Random seed', min_value=0, value=42, step=1)
    uploaded = st.file_uploader('Import program inputs', type=['csv','xlsx'])
    st.markdown('**Probability format:** use 0 to 1, for example 0.65 for 65%. Booked programs are automatically treated as 100%.')

if uploaded is not None:
    try:
        if uploaded.name.lower().endswith('.xlsx'):
            source = pd.read_excel(uploaded, engine='openpyxl')
        else:
            source = pd.read_csv(uploaded)
        missing = [c for c in REQUIRED if c not in source.columns]
        if missing:
            st.error('Missing columns: ' + ', '.join(missing))
            source = DEFAULT.copy()
    except Exception as exc:
        st.error(f'Could not read upload: {exc}')
        source = DEFAULT.copy()
else:
    source = DEFAULT.copy()

if 'programs' not in st.session_state or uploaded is not None:
    st.session_state.programs = source.copy()

edited = st.data_editor(
    st.session_state.programs,
    num_rows='dynamic',
    use_container_width=True,
    hide_index=True,
    column_config={
        'Status': st.column_config.SelectboxColumn(options=['Booked','Bid'], required=True),
        'Amount': st.column_config.NumberColumn(min_value=0.0, format='$%.2f', required=True),
        'Start Date': st.column_config.DateColumn(format='YYYY-MM-DD', required=True),
        'End Date': st.column_config.DateColumn(format='YYYY-MM-DD', required=True),
        'Win Probability': st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.01, format='%.2f'),
    },
    key='program_editor'
)
st.session_state.programs = edited


def validate(df):
    d = df.copy()
    for c in REQUIRED:
        if c not in d.columns:
            raise ValueError(f'Missing required column: {c}')
    d = d.dropna(subset=['Program','Status','Amount','Start Date','End Date'])
    d['Program'] = d['Program'].astype(str).str.strip()
    d['Status'] = d['Status'].astype(str).str.strip().str.title()
    d['Amount'] = pd.to_numeric(d['Amount'], errors='coerce')
    d['Start Date'] = pd.to_datetime(d['Start Date'], errors='coerce')
    d['End Date'] = pd.to_datetime(d['End Date'], errors='coerce')
    d['Win Probability'] = pd.to_numeric(d['Win Probability'], errors='coerce').fillna(0).clip(0,1)
    d['Funding Agency'] = d['Funding Agency'].fillna('Unspecified').astype(str)
    d.loc[d['Status'].eq('Booked'),'Win Probability'] = 1.0
    if d.empty: raise ValueError('Add at least one complete program.')
    if d[['Amount','Start Date','End Date']].isna().any().any(): raise ValueError('Amounts and dates must be valid.')
    if (~d['Status'].isin(['Booked','Bid'])).any(): raise ValueError('Status must be Booked or Bid.')
    if (d['Amount'] < 0).any(): raise ValueError('Amounts cannot be negative.')
    if (d['End Date'] < d['Start Date']).any(): raise ValueError('Each end date must be on or after its start date.')
    return d.reset_index(drop=True)


def monthly_matrix(df):
    start_month = df['Start Date'].dt.to_period('M').min()
    end_month = df['End Date'].dt.to_period('M').max()
    months = pd.period_range(start_month, end_month, freq='M')
    matrix = np.zeros((len(df), len(months)), dtype=float)
    for i,row in df.iterrows():
        first, last = row['Start Date'].to_period('M'), row['End Date'].to_period('M')
        mask = (months >= first) & (months <= last)
        matrix[i, mask] = row['Amount'] / int(mask.sum())
    return months, matrix


def run_simulation(df, n_runs, random_seed):
    months, allocation = monthly_matrix(df)
    booked_mask = df['Status'].eq('Booked').to_numpy()
    bid_idx = np.where(df['Status'].eq('Bid').to_numpy())[0]
    rng = np.random.default_rng(int(random_seed))
    inclusion = np.tile(booked_mask, (int(n_runs),1))
    if len(bid_idx):
        probs = df.loc[bid_idx,'Win Probability'].to_numpy()
        inclusion[:,bid_idx] = rng.random((int(n_runs),len(bid_idx))) < probs
    revenues = inclusion.astype(float) @ allocation
    totals = revenues.sum(axis=1)

    if len(bid_idx):
        packed = np.packbits(inclusion[:,bid_idx], axis=1)
        keys = [row.tobytes() for row in packed]
        counts = pd.Series(keys).value_counts()
        modal_key, modal_count = counts.index[0], int(counts.iloc[0])
        modal_run = keys.index(modal_key)
    else:
        modal_count, modal_run = int(n_runs), 0

    summary = pd.DataFrame({
        'Month': months.to_timestamp(),
        'Most Likely Specific Scenario': revenues[modal_run],
        'Average Scenario': revenues.mean(axis=0),
        'P10': np.percentile(revenues,10,axis=0),
        'P50': np.percentile(revenues,50,axis=0),
        'P90': np.percentile(revenues,90,axis=0),
    })
    won = df.loc[inclusion[modal_run] & df['Status'].eq('Bid').to_numpy(),'Program'].tolist()
    return summary, totals, inclusion, modal_count/int(n_runs), won, allocation

if st.button('Run simulation', type='primary', use_container_width=True):
    try:
        clean = validate(edited)
        summary, totals, inclusion, mode_freq, won, allocation = run_simulation(clean, runs, seed)
        st.session_state.result = (clean, summary, totals, inclusion, mode_freq, won, allocation)
    except Exception as exc:
        st.error(str(exc))

if 'result' in st.session_state:
    clean, summary, totals, inclusion, mode_freq, won, allocation = st.session_state.result
    booked_total = clean.loc[clean['Status'].eq('Booked'),'Amount'].sum()
    modal_total = summary['Most Likely Specific Scenario'].sum()
    average_total = totals.mean()
    p10, median, p90 = np.percentile(totals,[10,50,90])

    a,b,c,d = st.columns(4)
    a.metric('Most likely scenario total', f'${modal_total:,.0f}', f'{mode_freq:.1%} of runs')
    b.metric('Average scenario total', f'${average_total:,.0f}', 'Mean across runs')
    c.metric('Median total', f'${median:,.0f}', '50th percentile')
    d.metric('P10 to P90 total', f'${p10:,.0f} – ${p90:,.0f}')

    tab1, tab2, tab3, tab4 = st.tabs(['Monthly forecast','Most likely scenario','Funding agencies','Data export'])
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=summary['Month'], y=summary['P90'], line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=summary['Month'], y=summary['P10'], fill='tonexty', fillcolor='rgba(14,165,233,.16)', line=dict(width=0), name='P10–P90 band'))
        fig.add_trace(go.Scatter(x=summary['Month'], y=summary['Average Scenario'], name='Average scenario', line=dict(color='#10b981',width=3)))
        fig.add_trace(go.Scatter(x=summary['Month'], y=summary['Most Likely Specific Scenario'], name='Most likely specific scenario', line=dict(color='#0ea5e9',width=3)))
        fig.update_layout(height=520, yaxis_title='Revenue', xaxis_title='', hovermode='x unified', legend_orientation='h', template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
        display = summary.copy(); display['Month'] = display['Month'].dt.strftime('%Y-%m')
        st.dataframe(display.style.format({c:'${:,.0f}' for c in display.columns if c!='Month'}), use_container_width=True, hide_index=True)

    with tab2:
        st.subheader('Most likely specific scenario')
        st.write(f'This exact bid-win combination occurred in **{mode_freq:.1%}** of simulation runs.')
        st.write('**Booked programs included:** ' + (', '.join(clean.loc[clean.Status.eq('Booked'),'Program']) or 'None'))
        st.write('**Bid programs won:** ' + (', '.join(won) if won else 'None'))
        st.info('The most likely specific scenario is the exact combination of bid wins and losses seen most often. It is not necessarily the scenario closest to the average revenue.')

    with tab3:
        agency = clean.groupby('Funding Agency', as_index=False).apply(lambda g: pd.Series({
            'Booked Amount': g.loc[g.Status.eq('Booked'),'Amount'].sum(),
            'Probability-Weighted Bid Amount': (g.loc[g.Status.eq('Bid'),'Amount'] * g.loc[g.Status.eq('Bid'),'Win Probability']).sum()
        }), include_groups=False).reset_index(drop=True)
        st.bar_chart(agency.set_index('Funding Agency'))
        st.dataframe(agency.style.format({'Booked Amount':'${:,.0f}','Probability-Weighted Bid Amount':'${:,.0f}'}), use_container_width=True, hide_index=True)

    with tab4:
        csv = summary.copy(); csv['Month'] = csv['Month'].dt.strftime('%Y-%m')
        st.download_button('Download monthly forecast CSV', csv.to_csv(index=False).encode(), 'monthly_revenue_forecast.csv', 'text/csv')
        st.download_button('Download cleaned program inputs CSV', clean.to_csv(index=False).encode(), 'program_inputs.csv', 'text/csv')
        metadata = {'runs':int(runs),'seed':int(seed),'most_likely_frequency':mode_freq,'most_likely_bid_wins':won,'average_total':float(average_total),'p10_total':float(p10),'median_total':float(median),'p90_total':float(p90)}
        st.download_button('Download simulation metadata JSON', json.dumps(metadata,indent=2).encode(), 'simulation_metadata.json', 'application/json')

    with st.expander('Model assumptions and interpretation'):
        st.markdown('''
- Booked programs are included in every run.
- Each bid is sampled independently at its entered win probability.
- Each program amount is allocated evenly across every calendar month touched by its start and end dates, inclusive.
- The average scenario is the month-level arithmetic mean over all Monte Carlo runs.
- P10 and P90 represent uncertainty from bid outcomes only. Schedule slips, funding reductions, correlated awards, cost phasing, option years, and payment delays are not modeled.
''')
