#!/usr/bin/env python
"""
mlflow_tools.py - look at your mlflow runs from the terminal, no web browser needed.

Run it from anywhere with the AI_RTP environment active (it finds KBP_Project/mlruns next to this file):

    python mlflow_tools.py list                          # table of all runs
    python mlflow_tools.py list --sort best_val_loss     # ... best run first
    python mlflow_tools.py watch                         # follow the newest run until it finishes
    python mlflow_tools.py watch attn_unet3d_ep10 --once # print what a run has logged so far, then exit
    python mlflow_tools.py compare                       # all FINISHED runs side by side + PNG of the curves
                                                         # (3 panels: loss | dose score | DVH score)
    python mlflow_tools.py compare unet3d_ep10 attn_unet3d_ep10 --csv compare.csv
    python mlflow_tools.py artifacts unet3d_ep10         # files of a run and where they are on disk

A "run" can be given by its name (--run_name of the training script) or by its run id.
Defaults can be changed with --tracking_uri and --experiment (they go BEFORE the subcommand):
    python mlflow_tools.py --experiment my_experiment list

How it works (short): mlflow.search_runs() returns one table row per run (params, final metrics, status);
MlflowClient().get_metric_history() returns every logged (step, value) of one metric - that is what we need
for the learning curves, because search_runs() only keeps the LAST value of each metric.
"""
import argparse
import os
import sys
import time
import urllib.parse
from datetime import datetime

import matplotlib
matplotlib.use('Agg')                      # draw into files, no display needed (SSH friendly)
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_URI = 'file:' + os.path.join(HERE, 'mlruns')
DEFAULT_EXPERIMENT = 'RTOG_IMRT_teaching'
DEFAULT_PNG = os.path.join(HERE, 'results', 'rtog_imrt_teaching', 'compare_runs.png')

# A run whose status is RUNNING but that logged nothing for this long was most likely killed
# (mlflow only sets FAILED/FINISHED when the Python process ends normally; kill -9, SSH drop or a
# crashed GPU job leave "RUNNING" behind forever).  Can be changed with --stale_minutes.
STALE_MINUTES = 30

# Metrics shown by `compare` (final values; the test_* ones only exist after STEP 7 of the training script).
# test_dose_score_mean / test_DVH_score_mean are the two OpenKBP-style scores (see STEP 5 of the training script);
# runs made before they existed simply show an empty cell.
COMPARE_METRICS = [
    'test_dose_score_mean', 'test_DVH_score_mean',
    'best_val_loss', 'best_epoch',
    'test_MAE_all_mean', 'test_MAE_ptv_mean',
    'test_PTV_D95_absdiff_mean', 'test_PTV_D2_absdiff_mean', 'test_PTV_Mean_absdiff_mean',
    'test_Lungs_V20_absdiff_mean', 'test_Lungs_V5_absdiff_mean', 'test_Lungs_Mean_absdiff_mean',
    'test_Heart_V30_absdiff_mean', 'test_Heart_V5_absdiff_mean', 'test_Heart_Mean_absdiff_mean',
    'test_Esophagus_Mean_absdiff_mean', 'test_Spinalcord_Max_absdiff_mean',
]


# -------------------------------------------------------------------------------------------------------------
# helpers
# -------------------------------------------------------------------------------------------------------------
def connect(args):
    """Return (client, experiment) for the chosen tracking store; exit with a clear message if missing."""
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()
    exp = client.get_experiment_by_name(args.experiment)
    if exp is None:
        sys.exit('Experiment "{}" not found in {}.\n  Is the path right? Did you start a training run yet? '
                 'Use --tracking_uri / --experiment.'.format(args.experiment, args.tracking_uri))
    return client, exp


def load_runs_table(exp):
    """All runs of the experiment as a pandas DataFrame, oldest first.

    Columns: run_id, status, start_time, end_time, params.<name>, metrics.<name>, tags.mlflow.runName ...
    (metrics.<name> holds the LAST logged value; it is NaN when that metric was never logged).
    """
    df = mlflow.search_runs(experiment_ids=[exp.experiment_id], output_format='pandas')
    if df.empty:
        return df
    df = df.rename(columns={'tags.mlflow.runName': 'run_name'})
    local = datetime.now().astimezone().tzinfo                      # show local time
    for c in ('start_time', 'end_time'):                            # end_time is all-NaT while only RUNNING runs exist
        df[c] = pd.to_datetime(df[c], utc=True).dt.tz_convert(local)
    return df.sort_values('start_time').reset_index(drop=True)


def history(client, run_id, metric):
    """Every logged value of one metric as a DataFrame with columns step, value, time (ms), sorted by step."""
    h = client.get_metric_history(run_id, metric)
    rows = [{'step': m.step, 'value': m.value, 'time': m.timestamp} for m in h]
    return pd.DataFrame(rows, columns=['step', 'value', 'time']).sort_values('step').reset_index(drop=True)


def last_update_seconds(client, run_id, start_time):
    """Seconds since the run last logged train_loss (or since it started, if no epoch is done yet)."""
    h = history(client, run_id, 'train_loss')
    t_last = h['time'].iloc[-1] / 1000.0 if len(h) else start_time.timestamp()
    return time.time() - t_last


def effective_status(client, row, stale_minutes):
    """mlflow status, but a RUNNING run that has been silent for too long is shown as RUNNING? (stale)."""
    if row['status'] == 'RUNNING' and last_update_seconds(client, row['run_id'], row['start_time']) > stale_minutes * 60:
        return 'RUNNING?'
    return row['status']


def fmt_duration(seconds):
    if pd.isna(seconds):
        return ''
    seconds = int(seconds)
    return '{}:{:02d}:{:02d}'.format(seconds // 3600, seconds % 3600 // 60, seconds % 60)


def run_duration(row):
    """Duration in seconds; a RUNNING run has no end_time, so it is measured up to now."""
    end = row['end_time'] if pd.notna(row['end_time']) else pd.Timestamp.now(tz=row['start_time'].tz)
    return (end - row['start_time']).total_seconds()


def run_label(row):
    return row['run_name'] if isinstance(row['run_name'], str) else row['run_id'][:8]


def find_run(client, exp, text):
    """Resolve a run name or run id to an mlflow Run.  Several runs with the same name -> the newest one."""
    if len(text) == 32:                                     # looks like a run id
        try:
            return client.get_run(text)
        except mlflow.exceptions.MlflowException:
            pass
    runs = client.search_runs([exp.experiment_id], filter_string="tags.mlflow.runName = '{}'".format(text),
                              order_by=['attribute.start_time DESC'])
    if not runs:
        names = sorted({r.info.run_name for r in client.search_runs([exp.experiment_id]) if r.info.run_name})
        sys.exit('No run named or with id "{}". Available run names: {}'.format(text, ', '.join(names) or '(none)'))
    if len(runs) > 1:
        print('note: {} runs are called "{}"; using the newest ({}).'.format(len(runs), text, runs[0].info.run_id[:8]))
    return runs[0]


def newest_run(client, exp):
    runs = client.search_runs([exp.experiment_id], order_by=['attribute.start_time DESC'], max_results=1)
    if not runs:
        sys.exit('The experiment "{}" has no runs yet.'.format(exp.name))
    return runs[0]


def fmt_num(x, digits=3):
    return '' if pd.isna(x) else '{:.{}f}'.format(x, digits)


# -------------------------------------------------------------------------------------------------------------
# list
# -------------------------------------------------------------------------------------------------------------
def cmd_list(args):
    client, exp = connect(args)
    df = load_runs_table(exp)
    if df.empty:
        sys.exit('No runs in experiment "{}" yet.'.format(exp.name))
    out = []
    for _, r in df.iterrows():
        n_epochs_done = len(history(client, r['run_id'], 'train_loss'))      # one train_loss point per epoch
        out.append({
            'run_name': run_label(r),
            'status': effective_status(client, r, args.stale_minutes),
            'start': r['start_time'].strftime('%m-%d %H:%M'),
            'duration': fmt_duration(run_duration(r)),
            'model': r.get('params.model'),
            'epochs': r.get('params.epochs'),
            'done': n_epochs_done,
            'train_loss': r.get('metrics.train_loss'),
            'val_loss': r.get('metrics.val_loss'),
            'val_dvh_score': r.get('metrics.val_dvh_score'),             # latest epoch: handy while a run is RUNNING
            'best_val_loss': r.get('metrics.best_val_loss'),
            'best_epoch': r.get('metrics.best_epoch'),
            'test_dose_score': r.get('metrics.test_dose_score_mean'),
            'test_DVH_score': r.get('metrics.test_DVH_score_mean'),
            'test_MAE_ptv': r.get('metrics.test_MAE_ptv_mean'),
            'start_ts': r['start_time'],                                  # only used for sorting
        })
    table = pd.DataFrame(out)
    if args.sort:
        key = {'name': 'run_name', 'start': 'start_ts'}.get(args.sort, args.sort)
        if key not in table.columns:
            sys.exit('--sort must be one of: ' + ', '.join(c for c in table.columns if c != 'start_ts') + ', name')
        table = table.sort_values(key, na_position='last')              # NaN (not finished yet) goes last
    shown = table.drop(columns='start_ts')
    for c in ['train_loss', 'val_loss', 'val_dvh_score', 'best_val_loss', 'test_dose_score', 'test_DVH_score', 'test_MAE_ptv']:
        shown[c] = shown[c].map(fmt_num)                # runs without the new scores (older runs) show an empty cell
    shown['best_epoch'] = shown['best_epoch'].map(lambda x: '' if pd.isna(x) else str(int(x)))
    shown['epochs'] = shown['epochs'].fillna('')
    shown['model'] = shown['model'].fillna('')
    print('experiment {} | store {} | losses in Gy'.format(exp.name, args.tracking_uri))
    print(shown.to_string(index=False))
    if (shown['status'] == 'RUNNING?').any():
        print('\nRUNNING? = status is RUNNING but nothing was logged for {} min: probably killed. '
              'See MLFLOW_GUIDE.md ("killed runs").'.format(args.stale_minutes))


# -------------------------------------------------------------------------------------------------------------
# watch
# -------------------------------------------------------------------------------------------------------------
def cmd_watch(args):
    client, exp = connect(args)
    run = find_run(client, exp, args.run) if args.run else newest_run(client, exp)
    run_id, name = run.info.run_id, run.info.run_name
    n_total = run.data.params.get('epochs', '?')
    print('watching run {} ({}), {} epochs planned. Ctrl-C stops watching (training keeps running).'.format(
        name, run_id[:8], n_total))
    print('{:>6s} {:>12s} {:>12s} {:>14s} {:>14s} {:>10s}'.format(
        'epoch', 'train_loss', 'val_loss', 'val_dose_score', 'val_dvh_score', 'lr'))
    last_printed = 0
    warned = False
    try:
        while True:
            tr, va, lr, vd, vh = (history(client, run_id, m) for m in
                                  ('train_loss', 'val_loss', 'lr', 'val_dose_score', 'val_dvh_score'))
            # everything after train_loss is logged within the same epoch; merge on step and print only complete new
            # epochs.  The score columns are empty for runs made before they existed (left merge -> NaN).
            merged = tr.rename(columns={'value': 'train'})[['step', 'train']].merge(
                va.rename(columns={'value': 'val'})[['step', 'val']], on='step').merge(
                vd.rename(columns={'value': 'dose'})[['step', 'dose']], on='step', how='left').merge(
                vh.rename(columns={'value': 'dvh'})[['step', 'dvh']], on='step', how='left').merge(
                lr.rename(columns={'value': 'lr'})[['step', 'lr']], on='step', how='left')
            for _, m in merged[merged['step'] > last_printed].iterrows():
                print('{:>6d} {:>12.4f} {:>12.4f} {:>14s} {:>14s} {:>10.1e}'.format(
                    int(m['step']), m['train'], m['val'], fmt_num(m['dose'], 4), fmt_num(m['dvh'], 4), m['lr']), flush=True)
                last_printed = int(m['step'])
            run = client.get_run(run_id)                      # refresh the status
            status = run.info.status
            if status != 'RUNNING':
                print('run is {} after {} epochs.'.format(status, last_printed))
                best = run.data.metrics.get('best_val_loss')
                if best is not None:
                    print('best_val_loss {:.4f} Gy at epoch {:d}'.format(best, int(run.data.metrics.get('best_epoch', 0))))
                for name in ('test_dose_score_mean', 'test_DVH_score_mean', 'test_MAE_all_mean'):
                    if run.data.metrics.get(name) is not None:
                        print('{} {:.4f} Gy'.format(name, run.data.metrics[name]))
                return
            age = time.time() - (tr['time'].iloc[-1] / 1000.0 if len(tr) else run.info.start_time / 1000.0)
            if args.once:
                print('run is RUNNING ({} epochs done, last log {:.0f} s ago).'.format(last_printed, age))
                return
            if age > args.stale_minutes * 60 and not warned:
                print('warning: status is RUNNING but nothing was logged for {:.0f} min - was the job killed?'.format(age / 60),
                      flush=True)
                warned = True
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print('\nstopped watching.')


# -------------------------------------------------------------------------------------------------------------
# compare
# -------------------------------------------------------------------------------------------------------------
def cmd_compare(args):
    client, exp = connect(args)
    df = load_runs_table(exp)
    if args.runs:
        ids = [find_run(client, exp, t).info.run_id for t in args.runs]
        df = df.set_index('run_id').loc[ids].reset_index()
    else:
        df = df[df['status'] == 'FINISHED']                  # unfinished runs have no test metrics
    if len(df) < 1:
        sys.exit('No runs to compare.')
    labels = [run_label(r) for _, r in df.iterrows()]
    labels = [l if labels.count(l) == 1 else '{}_{}'.format(l, rid[:6]) for l, rid in zip(labels, df['run_id'])]

    # --- table: rows = items, one column per run ---
    table = pd.DataFrame(index=[], columns=labels, dtype=object)
    table.loc['status'] = [effective_status(client, r, args.stale_minutes) for _, r in df.iterrows()]
    table.loc['duration'] = [fmt_duration(run_duration(r)) for _, r in df.iterrows()]
    param_cols = [c for c in df.columns if c.startswith('params.')]
    for c in param_cols:                                     # only the params that differ between the runs
        values = df[c].tolist()
        if len({str(v) for v in values}) > 1:
            table.loc['param ' + c[len('params.'):]] = ['' if pd.isna(v) else str(v) for v in values]
    for m in COMPARE_METRICS:
        col = 'metrics.' + m
        if col in df.columns:
            table.loc[m] = [fmt_num(v, 3) if m != 'best_epoch' else ('' if pd.isna(v) else str(int(v))) for v in df[col]]
    print(table.to_string())
    print('\n(params shown only if they differ; empty = not logged yet, e.g. run still RUNNING; MAE and absdiff in Gy or % volume;'
          ' lower is better)')
    if args.csv:
        table.to_csv(args.csv)
        print('table saved to', args.csv)

    # --- figure: 3 panels (loss | dose score | DVH score), train (solid) and val (dashed), one colour per run ---
    panels = [('Loss: MAE, all voxels', 'train_loss', 'val_loss'),
              ('Dose score: MAE inside the body', 'train_dose_score', 'val_dose_score'),
              ('DVH score: OpenKBP-style', 'train_dvh_score', 'val_dvh_score')]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    colors = plt.get_cmap('tab10').colors
    skipped = []
    for ax, (title, m_train, m_val) in zip(axes, panels):
        for i, (label, rid) in enumerate(zip(labels, df['run_id'])):
            tr, va = history(client, rid, m_train), history(client, rid, m_val)
            if tr.empty and va.empty:                     # run made before this metric existed: nothing to draw
                skipped.append((label, title.split(':')[0]))
                continue
            color = colors[i % len(colors)]
            ax.plot(tr['step'], tr['value'], '-', color=color, label=label + ' train')
            ax.plot(va['step'], va['value'], '--', color=color, label=label + ' val')
        ax.set_xlabel('epoch')
        ax.set_ylabel('Gy')
        ax.set_title(title, loc='left', fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle('Learning curves (solid = train, dashed = validation; lower is better)', x=0.01, ha='left')
    for label, panel in skipped:
        print('note: run {} has no {} history (made before that metric was logged); left out of that panel.'.format(label, panel))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print('curves saved to', args.out)


# -------------------------------------------------------------------------------------------------------------
# artifacts
# -------------------------------------------------------------------------------------------------------------
def list_files(client, run_id, path=None):
    """All files below `path` (recursive); artifacts are listed one folder at a time by mlflow."""
    for info in client.list_artifacts(run_id, path):
        if info.is_dir:
            yield from list_files(client, run_id, info.path)
        else:
            yield info


def cmd_artifacts(args):
    client, exp = connect(args)
    run = find_run(client, exp, args.run)
    uri = run.info.artifact_uri
    if not uri.startswith('file:'):
        sys.exit('artifact store is not a local folder ({}); use "mlflow artifacts download".'.format(uri))
    root = urllib.parse.unquote(urllib.parse.urlparse(uri).path)         # file:///a/b -> /a/b
    print('run {} ({}) - artifact folder: {}'.format(run.info.run_name, run.info.run_id[:8], root))
    files = [f for f in list_files(client, run.info.run_id) if args.grep in f.path]
    for f in files:
        print('{:>9.1f} MB  {}'.format(f.file_size / 1e6, os.path.join(root, f.path)))
    print('{} files, {:.1f} MB'.format(len(files), sum(f.file_size for f in files) / 1e6))
    print('open a PNG with any viewer, or copy to your laptop:  scp "<user>@<server>:{}/learning_curves.png" .'.format(root))


# -------------------------------------------------------------------------------------------------------------
# command line
# -------------------------------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description='Inspect mlflow runs without the web UI.',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--tracking_uri', default=DEFAULT_URI, help='mlflow tracking store')
    p.add_argument('--experiment', default=DEFAULT_EXPERIMENT, help='experiment name')
    p.add_argument('--stale_minutes', type=float, default=STALE_MINUTES,
                   help='a RUNNING run silent for this long is flagged as probably killed')
    sub = p.add_subparsers(dest='command', required=True)

    s = sub.add_parser('list', help='table of all runs')
    s.add_argument('--sort', help='column to sort by, e.g. best_val_loss, test_MAE_ptv, name (default: start time)')
    s.set_defaults(func=cmd_list)

    s = sub.add_parser('watch', help='follow a run live')
    s.add_argument('run', nargs='?', help='run name or run id (default: most recent run)')
    s.add_argument('--interval', type=float, default=30, help='seconds between checks')
    s.add_argument('--once', action='store_true', help='print the epochs so far and exit')
    s.set_defaults(func=cmd_watch)

    s = sub.add_parser('compare', help='side-by-side table + PNG of the curves')
    s.add_argument('runs', nargs='*', help='run names or ids (default: all FINISHED runs)')
    s.add_argument('--out', default=DEFAULT_PNG, help='PNG with the overlaid curves')
    s.add_argument('--csv', help='also save the table as CSV')
    s.set_defaults(func=cmd_compare)

    s = sub.add_parser('artifacts', help='files of a run and their paths on disk')
    s.add_argument('run', help='run name or run id')
    s.add_argument('--grep', default='', help='only files whose path contains this text, e.g. .png')
    s.set_defaults(func=cmd_artifacts)

    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
