"""Generate every numeric table body and text macro of the manuscript from the
released result files. Usage: python make_tables.py <release_root>"""
import json,sys,os,itertools,re
ROOT=sys.argv[1]; R=os.path.join(ROOT,'results')
def J(p): return json.load(open(os.path.join(R,p)))
DS=['FD001','FD002','FD003','FD004']; BB=['LSTM','Transformer']
MECH=[('MSE_fixed','Fixed variance'),('NLL','Heteroscedastic'),('MC_Dropout_fixed','MC Dropout'),('Deep_Ensemble','Deep ensemble'),('CP_norm','Split-CP (norm)')]
M={}   # text macros
def mac(name,val): M[name]=val
def pct(x,d=1): return f"{100*x:.{d}f}"
def W(name,body): open(f'tables/{name}.tex','w').write(body)
# ---------- Table I
t1=J('clean/table1_2x2_summary.json')
rows=[]
for ds in ['FD001','FD002','FD004']:
    x=t1[ds]; c=x['contrasts']
    for k,(met,lab,fmt) in enumerate([('rmse','RMSE','{:.2f}'),('score','Score','{:.1f}')]):
        cells=[fmt.format(x[q][met]) for q in ['T_W','V_W','V_F','T_F']]
        sel=(x['T_W'][met]+x['T_F'][met])/2-(x['V_W'][met]+x['V_F'][met])/2
        nrm=(x['T_W'][met]+x['V_W'][met])/2-(x['T_F'][met]+x['V_F'][met])/2
        itr=(x['T_W'][met]-x['T_F'][met])-(x['V_W'][met]-x['V_F'][met])
        pre=f"\\multirow{{2}}{{*}}{{{ds}}} & " if k==0 else " & "
        rows.append(pre+f"{lab} & "+" & ".join(cells)+f" & ${sel:+.2f}$ & ${nrm:+.2f}$ & ${itr:+.2f}$ \\\\" if met=='rmse' else pre+f"{lab} & "+" & ".join(cells)+f" & ${sel:+.1f}$ & ${nrm:+.1f}$ & ${itr:+.1f}$ \\\\")
    rows.append("\\midrule")
W('tab_leak',"\n".join(rows[:-1])+"\n")
tot_rmse=[100*(t1[d]['V_F']['rmse']/t1[d]['T_W']['rmse']-1) for d in ['FD001','FD002','FD004']]
tot_score=[100*(t1[d]['V_F']['score']/t1[d]['T_W']['score']-1) for d in ['FD001','FD002','FD004']]
mac('LeakRMSE',f"{min(tot_rmse):.0f}--{max(tot_rmse):.0f}"); mac('LeakScore',f"{min(tot_score):.0f}--{max(tot_score):.0f}")
selonly=[100*(t1[d]['V_W']['rmse']/t1[d]['T_W']['rmse']-1) for d in ['FD001','FD002','FD004']]
mac('SelRMSE',f"{min(selonly):.0f}--{max(selonly):.0f}")
selscore=[100*(t1[d]['V_W']['score']/t1[d]['T_W']['score']-1) for d in ['FD001','FD002','FD004']]
mac('SelScoreMax',f"{max(selscore):.0f}")
nrmonly=[100*(t1[d]['V_F']['rmse']/t1[d]['V_W']['rmse']-1) for d in ['FD001','FD002','FD004']]
mac('NormRMSE',f"{min(nrmonly):.0f}--{max(nrmonly):.0f}")
dpicp=[t1[d]['V_W']['picp']-t1[d]['T_W']['picp'] for d in ['FD001','FD002','FD004']]
mac('SelPICP',", ".join(f"${v:+.2f}$" for v in dpicp))
mac('EcePooledFDone',f"{t1['FD001']['V_F']['ece']:.3f}")
# ---------- Table II
t2=J('clean/table2_clean_full.json'); fc=J('clean/fair_calibration_main_table.json')
def cl(bb,ds,m):
    if m in ('MSE_fixed','MC_Dropout_fixed'):
        r=fc[bb][ds][m]; return (r['picp_mean'],r['mpiw_mean'],r['ece_mean'],r['per_engine_compliance'],r['interval_score_mean'],r['wis_mean'])
    r=t2[bb][ds][m]; return (r['picp'],r['mpiw'],r['ece'],r['per_engine'],r['is'],r['wis'])
rows=[]
for ds in DS:
    for k,(m,lab) in enumerate(MECH):
        v=[f"{a:.3f} & {b:.2f} & {c:.3f} & {d:.2f} & {e:.1f}" for (a,b,c,d,e,_) in [cl(bb,ds,m) for bb in BB]]
        rows.append((f"\\multirow{{5}}{{*}}{{{ds}}}" if k==0 else "")+f" & {lab} & "+" & ".join(v)+" \\\\")
    rows.append("\\midrule")
W('tab_clean',"\n".join(rows[:-1])+"\n")
# IS rankings
ens_best=sum(1 for bb in BB for ds in DS if min((cl(bb,ds,m)[4],m) for m,_ in MECH)[1]=='Deep_Ensemble')
fix_worst=sum(1 for bb in BB for ds in DS if max((cl(bb,ds,m)[4],m) for m,_ in MECH)[1]=='MSE_fixed')
mac('EnsBestIS',str(ens_best)); mac('FixWorstIS',str(fix_worst))
wis_top_same=sum(1 for bb in BB for ds in DS if min((cl(bb,ds,m)[4],m) for m,_ in MECH)[1]==min((cl(bb,ds,m)[5],m) for m,_ in MECH)[1])
mac('WisTopSame',str(wis_top_same))
recov=[(cl('LSTM',ds,'MSE_fixed')[4]-cl('LSTM',ds,'NLL')[4])/(cl('LSTM',ds,'MSE_fixed')[4]-cl('LSTM',ds,'Deep_Ensemble')[4]) for ds in DS]
mac('HeadRecover',f"{100*sum(recov)/4:.0f}"); mac('HeadRecoverRange',f"{100*min(recov):.0f}--{100*max(recov):.0f}")
mcd_ge=sum(1 for bb in BB for ds in DS if cl(bb,ds,'MC_Dropout_fixed')[0]>=cl(bb,ds,'NLL')[0])
mac('McdCovGE',str(mcd_ge))
# ---------- Table III same split
S=J('seeds/samesplit_ensemble_control.json')
rows=[];gains=[];diffs=[]
for bb in BB:
    for ds in DS:
        x=S[bb][ds]; e=x['ensemble_same_split']
        cross=cl(bb,ds,'Deep_Ensemble'); nll=cl(bb,ds,'NLL')
        gains.append(x['single_model_is_mean_across_inits']-e['is_mean']); diffs.append(abs(e['is_mean']-cross[4]))
        rows.append(f"{bb} & {ds} & {nll[4]:.1f} & {x['single_model_is_mean_across_inits']:.1f} $\\pm$ {x['single_model_is_std_across_inits']:.1f} & {e['is_mean']:.1f} & {cross[4]:.1f} & {e['picp']:.3f} & {cross[0]:.3f} \\\\")
W('tab_samesplit',"\n".join(rows)+"\n")
mac('SameGain',f"{min(gains):.1f}--{max(gains):.1f}"); mac('SameGainCount',str(sum(g>0 for g in gains))); mac('SameCrossMax',f"{max(diffs):.1f}")
mac('OnePartSingleFDone',f"{S['LSTM']['FD001']['single_model_is_mean_across_inits']:.1f}"); mac('FivePartSingleFDone',f"{cl('LSTM','FD001','NLL')[4]:.1f}")
# ---------- Attribution
H=J('attribution/attribution_lstm_exact_interp.json'); G=J('attribution/attribution_transformer_exact_interp.json'); Bt=J('attribution/attribution_bootstrap_by_order_exact_interp.json')
def share(D,ds,m,o): r=D[ds][m]; return -r[o]['sigma_effect']/(r['C00']-r['C11'])
def attr(D,bb,extra):
    L=[]
    for m,lab in [('NLL','Heteroscedastic'),('CP_norm','CP-norm')]:
        A=[f"${100*share(D,ds,m,'order_A_freeze_sigma_first'):+.1f}\\%$" for ds in DS]; Bo=[f"${100*share(D,ds,m,'order_B_freeze_mu_first'):+.1f}\\%$" for ds in DS]
        L.append(f"\\multirow{{2}}{{*}}{{{lab}}} & scale, mean first & "+" & ".join(A)+" \\\\"); L.append(" & scale, scale first & "+" & ".join(Bo)+" \\\\")
    L.append("\\midrule")
    L.append("\\multicolumn{2}{l}{Mean dominates (point estimate), both orders} & "+" & ".join('yes' if all(D[ds][m]['mean_dominant_both_orders'] for m in ['NLL','CP_norm']) else '/'.join('yes' if D[ds][m]['mean_dominant_both_orders'] else 'no' for m in ['NLL','CP_norm']) for ds in DS)+" \\\\")
    for o,lab in [('order_A_freeze_sigma_first','mean first'),('order_B_freeze_mu_first','scale first')]:
        L.append(f"\\multicolumn{{2}}{{l}}{{CI excludes zero, {lab} (het./CP)}} & "+" & ".join("/".join('yes' if Bt[bb][ds][m][o]['excludes_zero'] else 'no' for m in ['NLL','CP_norm']) for ds in DS)+" \\\\")
    return "\n".join(L+extra)+"\n"
W('tab_attr',attr(H,'LSTM',["\\multicolumn{2}{l}{Baseline clamp fraction} & 0.34 & 0.22 & 0.40 & 0.076 \\\\"]))
W('tab_attr_transformer',attr(G,'Transformer',[]))
pe=sum(D[ds][m]['mean_dominant_both_orders'] for D in (H,G) for ds in DS for m in ['NLL','CP_norm'])
ciA=sum(Bt[bb][ds][m]['order_A_freeze_sigma_first']['excludes_zero'] for bb in BB for ds in DS for m in ['NLL','CP_norm'])
ciB=sum(Bt[bb][ds][m]['order_B_freeze_mu_first']['excludes_zero'] for bb in BB for ds in DS for m in ['NLL','CP_norm'])
mac('DomPoint',str(pe)); mac('CIA',str(ciA)); mac('CIB',str(ciB)); mac('CITot',str(ciA+ciB))
exc=[(bb,ds,m,o,Bt[bb][ds][m][o]) for bb in BB for ds in DS for m in ['NLL','CP_norm'] for o in ['order_A_freeze_sigma_first','order_B_freeze_mu_first'] if not Bt[bb][ds][m][o]['excludes_zero']]
mac('CIExceptions',"; ".join(f"{'CP-norm' if m=='CP_norm' else 'the heteroscedastic head'} on the {bb} for {ds}, {'mean-first' if o.startswith('order_A') else 'scale-first'} order (interval $[{x['ci95_lo']:+.3f}, {x['ci95_hi']:+.3f}]$)" for bb,ds,m,o,x in exc) or "none")
inter=[(D[ds][m]['C11']-D[ds][m]['C10'])-(D[ds][m]['C01']-D[ds][m]['C00']) for D in (H,G) for ds in DS for m in ['NLL','CP_norm']]
mac('InterMax',f"{max(abs(v) for v in inter):.3f}"); mac('InterPos',str(sum(v>0 for v in inter)))
helpA=[(bb,ds,m) for bb,D in [('LSTM',H),('Transformer',G)] for ds in DS for m in ['NLL','CP_norm'] if share(D,ds,m,'order_A_freeze_sigma_first')<0]
helpB=[(bb,ds,m) for bb,D in [('LSTM',H),('Transformer',G)] for ds in DS for m in ['NLL','CP_norm'] if share(D,ds,m,'order_B_freeze_mu_first')<0]
avgharm=sum(1 for bb,D in [('LSTM',H),('Transformer',G)] for ds in DS for m in ['NLL','CP_norm'] if (share(D,ds,m,'order_A_freeze_sigma_first')+share(D,ds,m,'order_B_freeze_mu_first'))/2>0)
mac('HelpA',str(len(helpA))); mac('HelpB',str(len(helpB))); mac('AvgHarm',str(avgharm))
mac('HelpAList',", ".join(f"{bb}/{ds}/{'CP' if m=='CP_norm' else 'het.'}" for bb,ds,m in helpA) or "none")
mac('ShareFDoneLSTM',pct(share(H,'FD001','NLL','order_A_freeze_sigma_first'),0))
# ---------- Gaussian half-life counts
rl=J('degradation/lstm_relative_half_life.json'); rt=J('degradation/transformer_half_life.json')
cells=[(nm,ds,m,D[ds][m]['rel_co']) for nm,D in [('LSTM',rl),('Transformer',rt)] for ds in DS for m,_ in MECH]
v=[c for c in cells if c[3] is not None]
mac('HalfBelowOneSeven',str(sum(c[3]<0.017 for c in v)))
out=max(v,key=lambda c:c[3]); rest=[c for c in v if c is not out]
mac('HalfMaxRest',pct(max(c[3] for c in rest))); mac('HalfOutlier',f"{out[0]}/{out[1]}/{out[2]}"); mac('HalfOutlierVal',pct(out[3]))
thr=J('degradation/threshold_crossover_refinement.json'); tle=J('degradation/threshold_crossover_leftside_extension.json')
mac('ThrGrand',pct(thr['refined_crossover_grand_mean'])); 
pt=[x['crossover_frac'] for x in tle['per_trial_first_crossing_from_clean'] if x.get('crossover_frac') is not None]
mac('ThrTrial',f"{100*min(pt):.1f}--{100*max(pt):.1f}")
mac('ThrLeftMin',f"{100*min(x['points_used'][1][0] for x in tle['per_trial_first_crossing_from_clean']):.1f}")
# ---------- Table V structured degradation
hd=J('degradation/degradation_half_life_bias_gain_drift.json'); sw=J('degradation/degradation_sweep_bias_gain_drift.json')
def g(x):
    p=x['picp']
    while isinstance(p,dict): p=p.get('grand_mean',list(p.values())[0])
    return float(p)
def cell(bb,deg,ds):
    fo=sw[bb][deg][ds]['feat_oob']; lv=fo['5']; lvv=lv['grand_mean'] if isinstance(lv,dict) else lv
    v=hd[bb][deg][ds]['NLL']['rel_co']
    if v is None: return '$<10^{-4}$' if (deg=='drift' and lvv<1e-4) else 'n.r.'
    return '$>10$' if v>0.10 else f"{100*v:.2f}\\%"
rows=[]
for bb in BB:
    for k,deg in enumerate(['bias','drift','gain']):
        rows.append((f"\\multirow{{3}}{{*}}{{{bb}}} & " if k==0 else " & ")+deg.capitalize()+" & "+" & ".join(cell(bb,deg,ds) for ds in DS)+" \\\\")
    rows.append("\\midrule")
W('tab_types',"\n".join(rows[:-1])+"\n")
spreads=[];worst_head=0;n=0
for bb in BB:
    for deg in ['drift','bias','gain']:
        for ds in DS:
            vv={m:g(sw[bb][deg][ds][m]['5']) for m,_ in MECH}; n+=1
            spreads.append(max(vv.values())-min(vv.values())); worst_head+=min(vv,key=vv.get) in ('NLL','CP_norm')
mac('MechSpread',f"{100*min(spreads):.0f}--{100*max(spreads):.0f}"); mac('WorstHead',str(worst_head)); mac('WorstHeadN',str(n))
nr=[hd[bb][deg][ds][m]['rel_co'] for bb in BB for deg in ['bias','gain'] for ds in ['FD002','FD004'] for m,_ in MECH if hd[bb][deg][ds][m]['rel_co'] is not None]
mac('StructRange',f"{100*min(nr):.0f}--{100*max(nr):.0f}")
# ---------- Table VI controls
d=J('degradation/drift_controls.json'); d1=J('degradation/drift_fixed_endpoint_shuffle_control.json')
L=d['LSTM']['FD002']; T=d['Transformer']['FD002']
ramp={bb:g(sw[bb]['drift']['FD002']['NLL']['5']) for bb in BB}
rampf=sw['LSTM']['drift']['FD002']['feat_oob']['5']; rampf=rampf['grand_mean'] if isinstance(rampf,dict) else rampf
rows=[f"Ramp (per window) & {rampf:.3f} & {ramp['LSTM']:.3f} & {ramp['Transformer']:.3f} \\\\",
      f"Ramp (contin.\\ trajectory) & {L['continuous']['feat_oob']:.3f} & {L['continuous']['picp_NLL']:.3f} & {T['continuous']['picp_NLL']:.3f} \\\\",
      f"Bias, equal end value & {L['bias_equal']['feat_oob']:.3f} & {L['bias_equal']['picp_NLL']:.3f} & {T['bias_equal']['picp_NLL']:.3f} \\\\",
      f"Reversed ramp & {L['reverse']['feat_oob']:.3f} & {L['reverse']['picp_NLL']:.3f} & {T['reverse']['picp_NLL']:.3f} \\\\",
      f"Shuffled ramp & {L['shuffled']['feat_oob']:.3f} & {L['shuffled']['picp_NLL']:.3f} & {T['shuffled']['picp_NLL']:.3f} \\\\",
      f"Single-channel ramp ($s_9$) & {L['singlech']['feat_oob']:.1e} & {L['singlech']['picp_NLL']:.3f} & {T['singlech']['picp_NLL']:.3f} \\\\"]
W('tab_controls',"\n".join(rows)+"\n")
for k,nm in [('RampL',ramp['LSTM']),('RampT',ramp['Transformer'])]: mac(k,f"{nm:.2f}")
for key,src in [('Cont',"continuous"),('BiasEq',"bias_equal"),('Rev',"reverse"),('Shuf',"shuffled"),('Single',"singlech")]:
    mac(key+'L',f"{L[src]['picp_NLL']:.2f}"); mac(key+'T',f"{T[src]['picp_NLL']:.2f}")
mac('FixEndOneL',f"{d1['LSTM']['1']:.2f}"); mac('FixEndFiveL',f"{d1['LSTM']['5']:.2f}")
mac('FixEndT',f"{min(d1['Transformer'].values()):.2f}--{max(d1['Transformer'].values()):.2f}")
# ---------- Table VII decision
c=J('decision/maintenance_decision_one_sided.json'); v2=J('decision/maintenance_decision_rul_corrected_v2.json')
def dec(bb,ds,cond,lab):
    r=c[bb][ds][cond]['NLL']['20']; u=v2[bb][ds][cond]['NLL']['20']
    cap=u.get('true_rul_at_trigger_capped125'); unc=u.get('true_rul_at_trigger_uncapped'); exc=u.get('excess_over_L_uncapped')
    f=lambda z:'--' if z is None else f"{z:.1f}"
    return f"{lab} & {ds}/{bb} & {r['overall_unrecognised_rate']:.3f} ({r['conditional_unrecognised_rate']:.3f}) & {r['premature_rate']:.3f} & {f(cap)} / {f(unc)} & {f(exc)} \\\\"
rows=[dec('LSTM','FD002','clean','Clean'),dec('LSTM','FD002','bias5pct','Bias 5\\%'),dec('LSTM','FD002','drift5pct','Drift 5\\%'),dec('LSTM','FD004','drift5pct','Drift 5\\%'),dec('Transformer','FD002','drift5pct','Drift 5\\%'),dec('LSTM','FD001','drift5pct','Drift 5\\%'),dec('LSTM','FD003','drift5pct','Drift 5\\%')]
W('tab_decision',"\n".join(rows)+"\n")
mechs=[m for m,_ in MECH]
prem=[c[bb][ds]['drift5pct']['NLL'][L_]['premature_rate'] for bb in BB for ds in ['FD002','FD004'] for L_ in ['10','20','30']]
mac('DriftPrem',f"{100*min(prem):.0f}--{100*max(prem):.0f}")
unc=[v2[bb][ds]['drift5pct']['NLL']['20']['true_rul_at_trigger_uncapped'] for bb in BB for ds in ['FD002','FD004']]
cap=[v2[bb][ds]['drift5pct']['NLL']['20']['true_rul_at_trigger_capped125'] for bb in BB for ds in ['FD002','FD004']]
exc=[v2[bb][ds]['drift5pct']['NLL']['20']['excess_over_L_uncapped'] for bb in BB for ds in ['FD002','FD004']]
mac('DriftRULunc',f"{min(unc):.0f}--{max(unc):.0f}"); mac('DriftRULcap',f"{min(cap):.0f}--{max(cap):.0f}"); mac('DriftExcess',f"{sum(exc)/len(exc):.0f}")
b=c['LSTM']['FD002']['bias5pct']['NLL']['20']; mac('BiasUnrec',f"{b['overall_unrecognised_rate']:.3f}"); mac('BiasUnrecCond',f"{b['conditional_unrecognised_rate']:.2f}")
nz=[(bb,ds,m,c[bb][ds]['drift5pct'][m]['20']['overall_unrecognised_rate']) for bb in BB for ds in DS for m in mechs if c[bb][ds]['drift5pct'][m]['20']['overall_unrecognised_rate']>0]
mac('DriftUnrecNZ',str(len(nz)))
mac('DriftUnrecNZList',"; ".join(f"{ {'MSE_fixed':'fixed variance','NLL':'heteroscedastic','MC_Dropout_fixed':'MC Dropout','Deep_Ensemble':'ensemble','CP_norm':'CP-norm'}[m]} on {ds}, {bb}, {r:.4f}" for bb,ds,m,r in nz) or "none")
ens_low=sum(1 for ds in DS if min(mechs,key=lambda m:c['LSTM'][ds]['bias5pct'][m]['20']['overall_unrecognised_rate'])=='Deep_Ensemble')
mac('EnsLowUnrecLSTM',str(ens_low))
rev=top=0
for bb in BB:
  for ds in DS:
    for cond in ['clean','gaussian1pct','drift5pct','bias5pct']:
      for L_ in ['10','20','30']:
        o={r:tuple(sorted(mechs,key=lambda m:c[bb][ds][cond][m][L_]['cost_by_ratio'][r])) for r in ['5','20','100']}
        rev+=len(set(o.values()))>1; top+=len({x[0] for x in o.values()})>1
mac('Rev',str(rev)); mac('RevTop',str(top))
sig=[sum(1 for x in c[bb][ds]['_bootstrap_top_vs_others'].values() if x['significant']) for bb in BB for ds in DS]
mac('BootSig',f"{min(sig)}--{max(sig)}")
fd1=c['LSTM']['FD001']['drift5pct']['NLL']['20']['premature_rate']; fd3=c['LSTM']['FD003']['drift5pct']['NLL']['20']['premature_rate']
mac('PremFDone',pct(fd1)); mac('PremFDthree',pct(fd3,0))
# ---------- diagnostics A1/A2
a1=J('diagnostics/A1_perturbation_target_diagnostic.json'); a2=J('diagnostics/A2_target_alignment_diagnostic.json')
ratio=[a1[ds][bb]['sensors']['drift5pct']['premature_rate_L20_grand_mean']/a1[ds][bb]['joint']['drift5pct']['premature_rate_L20_grand_mean'] for ds in ['FD002','FD004'] for bb in BB]
mac('SensJointRatio',f"{min(ratio):.2f}--{max(ratio):.2f}")
setonly=[a1[ds][bb]['settings']['drift5pct']['premature_rate_L20_grand_mean'] for ds in ['FD002','FD004'] for bb in BB]
mac('SetOnlyPrem',f"{100*min(setonly):.0f}--{100*max(setonly):.0f}")
sens=[a1[ds][bb]['sensors']['drift5pct']['premature_rate_L20_grand_mean'] for ds in ['FD002','FD004'] for bb in BB]
mac('SensPrem',f"{100*min(sens):.0f}--{100*max(sens):.0f}")
dR=[a2[ds][bb]['clean']['RULminus1_train_convention']['rmse']-a2[ds][bb]['clean']['official_RUL_current']['rmse'] for ds in DS for bb in BB]
dP=[a2[ds][bb]['clean']['RULminus1_train_convention']['picp']-a2[ds][bb]['clean']['official_RUL_current']['picp'] for ds in DS for bb in BB]
mac('AlignRMSE',f"{min(dR):+.2f} to {max(dR):+.2f}"); mac('AlignPICP',f"{min(dP):+.3f} to {max(dP):+.3f}")
with open('tables/numbers.tex','w') as f:
    for k,v in M.items(): f.write(f"\\newcommand{{\\N{k}}}{{{v}}}\n")
for k,v in M.items(): print(f"{k:20s} {v}")
# ---------- extra macros for text
# Table I every-cell-but check
worse=[]
for ds in ['FD001','FD002','FD004']:
    x=t1[ds]
    for met in ['rmse','score']:
        if x['T_W'][met]>x['V_W'][met]: worse.append((ds,met,'W'))
        if x['T_F'][met]>x['V_F'][met]: worse.append((ds,met,'F'))
M['LeakExceptions']="; ".join(f"{ds} {met} under {'fit-only' if n=='F' else 'whole-file'} normalisation" for ds,met,n in worse) or "none"
M['LeakExceptionsN']=str(len(worse))
# MCD vs head IS on LSTM FD002
M['McdISFDtwo']=f"{cl('LSTM','FD002','MC_Dropout_fixed')[4]:.1f}"; M['HeadISFDtwo']=f"{cl('LSTM','FD002','NLL')[4]:.1f}"
# CP deviations
dev=[(bb,ds,cl(bb,ds,'CP_norm')[0]) for bb in BB for ds in DS]
out=[x for x in dev if abs(x[2]-0.9)>0.03]
M['CPOutside']="; ".join(f"{p:.3f} (CP-norm, {bb}, {ds}; ${p-0.9:+.3f}$)" for bb,ds,p in out) or "none"
M['CPOutsideN']=str(len(out)); M['CPInsideN']=str(8-len(out))
# fair ratio
fs=J('clean/fair_calibration_sigma_fixed.json')
rat=[fs[bb][ds]['mean_ratio_calib_over_train'] for bb in BB for ds in DS]
M['FairRatio']=f"{100*(min(rat)-1):.0f}--{100*(max(rat)-1):.0f}"
# drift coverage per dataset (NLL, 5%)
dc={ds:[g(sw[bb]['drift'][ds]['NLL']['5']) for bb in BB] for ds in DS}
for ds in DS: M['DriftCov'+{'1':'One','2':'Two','3':'Three','4':'Four'}[ds[-1]]]=f"{min(dc[ds]):.2f}--{max(dc[ds]):.2f}"
def fo(bb,deg,ds):
    x=sw[bb][deg][ds]['feat_oob']['5']; return x['grand_mean'] if isinstance(x,dict) else x
M['DriftFoobSix']=f"{100*min(fo(bb,'drift',ds) for bb in BB for ds in ['FD002','FD004']):.0f}--{100*max(fo(bb,'drift',ds) for bb in BB for ds in ['FD002','FD004']):.0f}"
M['BiasFoobSix']=f"{100*min(fo(bb,'bias',ds) for bb in BB for ds in ['FD002','FD004']):.0f}--{100*max(fo(bb,'bias',ds) for bb in BB for ds in ['FD002','FD004']):.0f}"
M['DriftFoobSingleMax']=f"{max(fo(bb,'drift',ds) for bb in BB for ds in ['FD001','FD003']):.1e}"
dr=[hd[bb]['drift'][ds]['NLL']['rel_co'] for bb in BB for ds in ['FD002','FD004']]
M['DriftHalfSix']=f"{100*min(dr):.1f}--{100*max(dr):.1f}"
bg=[hd[bb][deg][ds]['NLL']['rel_co'] for bb in BB for deg in ['bias','gain'] for ds in ['FD002','FD004']]
M['BGHalfSix']=f"{100*min(bg):.0f}--{100*max(bg):.0f}"
_gv=[c_[3] for c_ in cells if c_[3] is not None]; M['NoiseMedian']=f"{100*sorted(_gv)[len(_gv)//2]:.1f}"
M['ContFoob']=f"{L['continuous']['feat_oob']:.3f}"
# premature at L=20 NLL, six-condition
p20=[c[bb][ds]['drift5pct']['NLL']['20']['premature_rate'] for bb in BB for ds in ['FD002','FD004']]
M['DriftPremTwenty']=f"{100*min(p20):.0f}--{100*max(p20):.0f}"
# A1 perturbation target
def a(ds,bb,arm,k): return a1[ds][bb][arm]['drift5pct'][k]
M['AOneFDtwoSens']=f"{a('FD002','LSTM','sensors','premature_rate_L20_grand_mean'):.2f}"; M['AOneFDtwoJoint']=f"{a('FD002','LSTM','joint','premature_rate_L20_grand_mean'):.2f}"
fd4=[(a('FD004',bb,'sensors','premature_rate_L20_grand_mean'),a('FD004',bb,'joint','premature_rate_L20_grand_mean'),a('FD004',bb,'settings','premature_rate_L20_grand_mean')) for bb in BB]
M['AOneFDfourSens']=f"{min(x[0] for x in fd4):.2f}--{max(x[0] for x in fd4):.2f}"; M['AOneFDfourJoint']=f"{min(x[1] for x in fd4):.2f}--{max(x[1] for x in fd4):.2f}"; M['AOneFDfourSet']=f"{min(x[2] for x in fd4):.2f}--{max(x[2] for x in fd4):.2f}"
setcov=[a(ds,bb,'settings','picp_grand_mean') for ds in ['FD002','FD004'] for bb in BB]
M['AOneSetCov']=f"{min(setcov):.2f}--{max(setcov):.2f}"
# Appendix D
es=J('seeds/ensemble_size_sweep_fd004_interval_score.json'); ir=J('seeds/ensemble_independent_replication.json')
M['MSweepPICP']=f"{es['2']['picp_mean']:.3f} to {es['10']['picp_mean']:.3f}"; M['MSweepIS']=f"{es['2']['is_mean']:.1f} to {es['10']['is_mean']:.1f}"
grp=lambda ds:"/".join(f"{x:.1f}" for x in (ir[ds]['groups'] if isinstance(ir[ds]['groups'][0],(int,float)) else [gg.get('is_mean',gg.get('IS',0)) for gg in ir[ds]['groups']]))
M['IndepIS']=", ".join(grp(ds) for ds in DS)
M['IndepSig']=", ".join(ds for ds in DS if ir[ds]['bootstrap_single_nll_minus_ensemble_group0'].get('ci95_lo',ir[ds]['bootstrap_single_nll_minus_ensemble_group0'].get('ci_lo',-1))>0) or "none"
nv=J('degradation/lstm_ensemble_n5_vs_n15_variance.json'); M['NFiveFifteenRaw']=str(list(nv.keys())[:4])
with open('tables/numbers.tex','w') as f:
    for k,v_ in M.items(): f.write(f"\\newcommand{{\\N{k}}}{{{v_}}}\n")
for k in ['LeakExceptions','McdISFDtwo','HeadISFDtwo','CPOutside','CPInsideN','FairRatio','DriftCovOne','DriftCovTwo','DriftCovThree','DriftCovFour','DriftFoobSix','BiasFoobSix','DriftFoobSingleMax','DriftHalfSix','BGHalfSix','NoiseMedian','ContFoob','DriftPremTwenty','AOneFDtwoSens','AOneFDtwoJoint','AOneFDfourSens','AOneFDfourJoint','AOneFDfourSet','AOneSetCov','MSweepPICP','MSweepIS','IndepIS','IndepSig','NFiveFifteenRaw']: print(f"{k:20s} {M[k]}")
dd=J('degradation/degradation_dose_response_overlap_summary.json')
mad=[dd[bb][ds][m]['mean_abs_deviation_picp'] for bb in dd for ds in dd[bb] for m in dd[bb][ds]]
M['DoseMAD']=f"{min(mad):.2f}--{max(mad):.2f}"
rest=[c_ for c_ in cells if c_[3] is not None and c_ is not max((c for c in cells if c[3] is not None),key=lambda c:c[3])]
mx=max(rest,key=lambda c:c[3]); lab={'MSE_fixed':'fixed-variance','NLL':'heteroscedastic','MC_Dropout_fixed':'MC Dropout','Deep_Ensemble':'ensemble','CP_norm':'CP-norm'}
M['HalfMaxRestCell']=f"the {lab[mx[2]]} {mx[0]} on {mx[1]}"
M['SelPICPsign']=""
M['TableIFDtwoScore']=f"{t1['FD002']['T_F']['score']:.1f} against {t1['FD002']['V_F']['score']:.1f}"
with open('tables/numbers.tex','w') as f:
    for k,v_ in M.items(): f.write(f"\\newcommand{{\\N{k}}}{{{v_}}}\n")
print(M['DoseMAD'],M['HalfMaxRestCell'],M['TableIFDtwoScore'])
nv=J('degradation/lstm_ensemble_n5_vs_n15_variance.json')
M['NFiveFifteen']=", ".join(f"{nv[ds]['picp']['std_ratio_15_over_5']:.2f}" for ds in ['FD001','FD002','FD004'])
with open('tables/numbers.tex','w') as f:
    for k,v_ in M.items(): f.write(f"\\newcommand{{\\N{k}}}{{{v_}}}\n")
ir=J('seeds/ensemble_independent_replication.json')
_bs=[ir[ds]['bootstrap_single_nll_minus_ensemble_group0'] for ds in DS]
M['IndepSigN']={0:'none',1:'one',2:'two',3:'three',4:'all four'}[sum(b['ci95_lo']>0 for b in _bs)]
M['IndepPosN']={0:'none',1:'one',2:'two',3:'three',4:'all four'}[sum(b['mean_diff']>0 for b in _bs)]
with open('tables/numbers.tex','w') as f:
    for k,v_ in M.items(): f.write(f"\\newcommand{{\\N{k}}}{{{v_}}}\n")
# --- controlled 2x2 effect macros (cycles / score units)
def eff(ds,met):
    v={q:t1[ds][q][met] for q in ['T_W','V_W','V_F','T_F']}
    return ((v['T_W']+v['T_F'])/2-(v['V_W']+v['V_F'])/2, (v['T_W']+v['V_W'])/2-(v['T_F']+v['V_F'])/2, (v['T_W']-v['T_F'])-(v['V_W']-v['V_F']))
D3=['FD001','FD002','FD004']
sr=[eff(d,'rmse')[0] for d in D3]; nr=[eff(d,'rmse')[1] for d in D3]; ir_=[eff(d,'rmse')[2] for d in D3]
ss=[eff(d,'score')[0] for d in D3]; ns=[eff(d,'score')[1] for d in D3]; is_=[eff(d,'score')[2] for d in D3]
M['SelEffRMSE']=f"{min(sr):.2f} to {max(sr):.2f}"; M['NormEffRMSE']=f"{min(nr):.2f} to {max(nr):.2f}"; M['IntEffRMSE']=f"{min(ir_):.2f} to {max(ir_):.2f}"
M['SelEffScore']=f"{min(ss):.0f} to {max(ss):.0f}"; M['NormEffScore']=f"{min(ns):.0f} to {max(ns):.0f}"; M['IntEffScore']=f"{min(is_):.0f} to {max(is_):.0f}"
M['NormAbsMaxRMSE']=f"{max(abs(x) for x in nr):.2f}"; M['SelAbsMaxRMSE']=f"{max(abs(x) for x in sr):.2f}"
M['LeakRMSEmax']=f"{max(100*(t1[d]['V_F']['rmse']/t1[d]['T_W']['rmse']-1) for d in D3):.0f}"
M['LeakScoremax']=f"{max(100*(t1[d]['V_F']['score']/t1[d]['T_W']['score']-1) for d in D3):.0f}"
with open('tables/numbers.tex','w') as f:
    for k,v_ in M.items(): f.write(f"\\newcommand{{\\N{k}}}{{{v_}}}\n")
for k in ['SelEffRMSE','NormEffRMSE','IntEffRMSE','SelEffScore','NormEffScore','IntEffScore','NormAbsMaxRMSE','LeakRMSEmax','LeakScoremax','LeakExceptions']: print(k,M[k])
# ---------- supplementary table bodies
Bt2=Bt
sup=[]
for bb,D in [("LSTM",H),("Transformer",G)]:
    for ds in DS:
        for m,lab in [('NLL','Heterosc.'),('CP_norm','CP-norm')]:
            r=D[ds][m]; A=r['order_A_freeze_sigma_first']; Bo=r['order_B_freeze_mu_first']; I=(r['C11']-r['C10'])-(r['C01']-r['C00'])
            sup.append(f"{bb} & {ds} & {lab} & {100*r['crossover_feat_oob_exact']:.2f} & {r['C00']:.3f} & {r['C10']:.3f} & {r['C01']:.3f} & {r['C11']:.3f} & {A['mu_effect']:+.4f} & {A['sigma_effect']:+.4f} & {Bo['mu_effect']:+.4f} & {Bo['sigma_effect']:+.4f} & {I:+.4f} & {'yes' if r['mean_dominant_both_orders'] else 'no'} \\\\")
W('supp_s1',"\n".join(sup)+"\n")
bl=[]
for bb in BB:
    for ds in DS:
        for m,lab in [('NLL','Heterosc.'),('CP_norm','CP-norm')]:
            xa=Bt2[bb][ds][m]['order_A_freeze_sigma_first']; xb=Bt2[bb][ds][m]['order_B_freeze_mu_first']
            bl.append(f"{bb} & {ds} & {lab} & {xa['mean']:+.3f} [{xa['ci95_lo']:+.3f}, {xa['ci95_hi']:+.3f}] & {'yes' if xa['excludes_zero'] else 'no'} & {xb['mean']:+.3f} [{xb['ci95_lo']:+.3f}, {xb['ci95_hi']:+.3f}] & {'yes' if xb['excludes_zero'] else 'no'} \\\\")
W('supp_boot',"\n".join(bl)+"\n")
labm={'NLL':'Heterosc.','MSE_fixed':'Fixed var.','MC_Dropout_fixed':'MC Drop.','Deep_Ensemble':'Ensemble','CP_norm':'CP-norm'}
conds=[('clean','Clean'),('gaussian1pct','Noise 1\\%'),('bias5pct','Bias 5\\%'),('drift5pct','Drift 5\\%')]
ml=['NLL','MSE_fixed','MC_Dropout_fixed','Deep_Ensemble','CP_norm']
lines=[];revs=[]
for bb in BB:
    for ds in DS:
        for ck,cll in conds:
            for L_ in ['10','20','30']:
                costs={m:c[bb][ds][ck][m][L_]['cost_by_ratio'] for m in ml}
                ranks={r_:{m:i+1 for i,m in enumerate(sorted(ml,key=lambda m:costs[m][r_]))} for r_ in ['5','20','100']}
                for m in ml:
                    r=c[bb][ds][ck][m][L_]; rul=r['true_rul_at_trigger']; rul='--' if rul is None else f"{rul:.1f}"
                    lines.append(f"{bb} & {ds} & {cll} & {L_} & {labm[m]} & {r['overall_unrecognised_rate']:.4f} / {r['conditional_unrecognised_rate']:.3f} & {r['premature_rate']:.4f} & {rul} & {costs[m]['5']:.3f} / {costs[m]['20']:.3f} / {costs[m]['100']:.3f} & {ranks['5'][m]}/{ranks['20'][m]}/{ranks['100'][m]} \\\\")
                o={r_:tuple(sorted(ml,key=lambda m:costs[m][r_])) for r_ in ['5','20','100']}
                revs.append(f"{bb} & {ds} & {cll} & {L_} & {'yes' if len(set(o.values()))>1 else 'no'} & {'yes' if len({x[0] for x in o.values()})>1 else 'no'} \\\\")
W('supp_s2',"\n".join(lines)+"\n"); W('supp_s2b',"\n".join(revs)+"\n")
W('supp_s2c',"\n".join(f"{bb} & {ds} & {sum(1 for x in c[bb][ds]['_bootstrap_top_vs_others'].values() if x['significant'])} / {len(c[bb][ds]['_bootstrap_top_vs_others'])} \\\\" for bb in BB for ds in DS)+"\n")
W('supp_s3',"\n".join(f"{bb} & {ds} & "+" & ".join(f"{cl(bb,ds,m)[5]:.2f}" for m in ml)+" \\\\" for bb in BB for ds in DS)+"\n")
lat=J('latency/lstm_latency.json')
def lt(kind):
    out=[]
    for ds in DS:
        for bb in BB:
            for bs in ['1','32','512']:
                cells=[]
                for m in ['NLL','MSE','CP_norm','MC_Dropout_T50','Deep_Ensemble_M5']:
                    x=lat[ds][bb][bs][m][kind]; k=int(bs)
                    cells.append(f"{x['median_ms']*1000/k:.2f} ({x['iqr_ms']*1000/k:.2f})")
                out.append(f"{ds} & {bb} & {bs} & "+" & ".join(cells)+" \\\\")
    return "\n".join(out)+"\n"
W('supp_s4_cuda',lt('cuda_event')); W('supp_s4_wall',lt('wallclock'))
import csv as _csv
pv=[r for r in _csv.DictReader(open(os.path.join(R,'table_provenance.csv'),encoding='utf-8')) if not r['table'].startswith('Joint-perturbation')]
def esc(x):
    x=x.replace('\\','/').replace('_','\\_').replace('%','\\%').replace('&','\\&').replace('#','\\#')
    return x.replace('\\_','\\_\\allowbreak ').replace('.py','\\allowbreak .py')
keys=['table','generating_script','labels','evaluation_unit','engine_equal_weighted','aggregation','noise_range_reference']
W('supp_s0',"\n".join(" & ".join(esc(r.get(k,'--')) for k in keys)+" \\\\" for r in pv)+"\n")
def repro_bit():
    # Read the bit-identical/total counts from REPRODUCE.md's own MD5
    # manifest, rather than hardcoding them here -- they must always match
    # what REPRODUCE.md actually documents and lists hashes for.
    repro_text = open(os.path.join(ROOT, 'REPRODUCE.md'), encoding='utf-8').read()
    m = re.search(r'## The (\d+) files and their MD5.*?```\n(.*?)```', repro_text, re.S)
    lines = [l for l in m.group(2).splitlines() if l.strip()]
    total = len(lines)
    latency_exempt = sum(1 for l in lines if 'latency-exempt' in l)
    return f"{total - latency_exempt} of the {total}"
M['ReproBit']=repro_bit()
with open('tables/numbers.tex','w') as f:
    for k,v_ in M.items(): f.write(f"\\newcommand{{\\N{k}}}{{{v_}}}\n")
print("supp tables written")
