# Superseded results

These three files are the pre-audit results referenced in the paper's
Appendix A ("Audit History"). They are kept for provenance only; the
current, corrected results are the ones under `results/{clean,degradation,
attribution,decision,latency,seeds}/`. Quoting Appendix A directly:

> The same audit found that the earlier MC Dropout baseline had omitted
> the aleatory variance term, producing coverage of 0.40--0.49, and that
> an earlier ensemble evaluation had given each member its own noise
> realisation, which averages the perturbation at 1/sqrt(M) and
> manufactured an apparent robustness. An earlier mean-only counterfactual
> used a separately trained MSE model and disagreed in sign with the
> frozen-output control on two of three sub-datasets.

- `mc_dropout_missing_aleatory_term.json` -- the earlier MC Dropout
  baseline described above (coverage 0.40--0.49 from omitting the
  aleatory-variance term when forming the predictive interval).
- `ensemble_independent_noise_per_member.json` -- the earlier ensemble
  evaluation described above (each member given its own independent noise
  realisation, which implicitly averages the perturbation at 1/sqrt(M)
  and manufactured an apparent robustness).
- `mse_proxy_mean_only_counterfactual.json` -- the earlier mean-only
  counterfactual described above (a separately trained MSE model used as
  a mean-only proxy, which disagreed in sign with the frozen-output
  control on two of three sub-datasets).

None of these three files are used to produce any number reported in the
current manuscript or supplementary material.

## checkpoints_tw_vw_uncontrolled/

Added in R9. The original T/W and V/W checkpoints for Table I (trained on
100% and 80% of the training-file engines respectively, rather than the
60% canonical `fit_units` split used by every other cell) -- see
`code/superseded/README.md` for why they were replaced by the controlled
T/W'/V/W' cells. Kept for provenance only; not read by
`code/build_table1.py`.
